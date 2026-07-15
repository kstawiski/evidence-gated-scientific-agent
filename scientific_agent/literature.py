"""Path-confined PubMed metadata and legitimate open-access full-text tools.

The model never supplies an arbitrary download URL.  Network access is restricted
to fixed NCBI/PMC endpoints, while a separate import tool can copy one regular PDF
from the managed browser download inbox after deterministic content checks.
"""

from __future__ import annotations

import hashlib
import ipaddress
import io
import json
import os
import re
import shutil
import stat
import tarfile
import threading
import time
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import httpx

from .config import LiteratureSettings, SandboxSettings
from .provenance import sha256_file
from .schemas import RetrievalEvidence, ScientificReport, SourceRecord


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PUBTATOR_URL = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
)
ALLOWED_DOWNLOAD_HOSTS = {"ftp.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov"}
ALLOWED_LITERATURE_HOSTS = {
    "eutils.ncbi.nlm.nih.gov",
    "www.ncbi.nlm.nih.gov",
    *ALLOWED_DOWNLOAD_HOSTS,
}
MAX_HTTP_REDIRECTS = 5
MAX_XML_BYTES = 24 * 1024 * 1024
MAX_SEARCH_RESULTS = 20
MIN_PDF_BYTES = 10_000
MIN_ARTICLE_TEXT_CHARS = 500
MAX_PDF_TEXT_BYTES = 16 * 1024 * 1024
MAX_ACQUIRED_MARKDOWN_BYTES = 24 * 1024 * 1024
MAX_AUDIT_ARTICLES = 8
MAX_AUDIT_PASSAGES_PER_ARTICLE = 4
MAX_AUDIT_PASSAGE_CHARS = 1_500
MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_PDF_CANDIDATES = 5
AUTOMATIC_OA_TERMS_WARNING = (
    "Automatically acquired from the PMC Open Access subset; reuse remains subject "
    "to the recorded article license."
)
MANUAL_PDF_TERMS_WARNING = (
    "User-provided private browser artifact; Evidence Bench does not grant reuse or "
    "redistribution rights. Verify publisher and institutional terms before export."
)
_PMID = re.compile(r"^[1-9][0-9]{0,8}$")
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,239}\.pdf$", re.I)
_CITEKEY = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
_WORD = re.compile(r"[a-z0-9]+")
_BIBLIOGRAPHY_LOCK = threading.Lock()


class LiteratureError(RuntimeError):
    """A deterministic acquisition or verification failure."""


def _metadata_artifact_for_source(
    source: SourceRecord,
    retrieval: RetrievalEvidence | None,
) -> Path:
    metadata_artifacts: dict[str, Path] = {}
    for value in retrieval.artifacts if retrieval else []:
        path = Path(value)
        if path.suffix.lower() == ".json" and path.parent.name == "metadata":
            metadata_artifacts[str(path.resolve())] = path

    candidates = {
        resolved: path
        for resolved, path in metadata_artifacts.items()
        if source.citekey and path.name == f"{source.citekey}.json"
    }
    if not candidates and source.pmid:
        for resolved, path in metadata_artifacts.items():
            try:
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.stat().st_size > 1024 * 1024
                ):
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                article = payload.get("article")
                if (
                    isinstance(article, dict)
                    and str(article.get("pmid")) == source.pmid
                ):
                    candidates[resolved] = path
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
    if len(candidates) != 1:
        raise LiteratureError(
            "PubMed source must map to exactly one retrieved acquisition metadata file"
        )
    path = next(iter(candidates.values()))
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise LiteratureError("PubMed acquisition metadata is missing or invalid")
    return path.resolve()


def load_acquired_article_record(
    source: SourceRecord,
    retrieval: RetrievalEvidence | None,
) -> tuple[dict[str, Any], Path, str]:
    """Load controller-recorded acquisition metadata and bounded local text."""

    metadata_path = _metadata_artifact_for_source(source, retrieval)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiteratureError("PubMed acquisition metadata is not valid JSON") from exc
    article = payload.get("article")
    acquisition = payload.get("acquisition")
    if not isinstance(article, dict) or not isinstance(acquisition, dict):
        raise LiteratureError("PubMed acquisition metadata has an invalid schema")
    markdown_value = acquisition.get("markdown_path")
    if not isinstance(markdown_value, str):
        raise LiteratureError("PubMed acquisition metadata has no Markdown path")
    markdown_path = Path(markdown_value)
    retrieval_paths = {
        str(Path(item).resolve()) for item in (retrieval.artifacts if retrieval else [])
    }
    if str(markdown_path.resolve()) not in retrieval_paths:
        raise LiteratureError("acquired Markdown is absent from retrieval evidence")
    if (
        not markdown_path.is_file()
        or markdown_path.is_symlink()
        or markdown_path.stat().st_size > MAX_ACQUIRED_MARKDOWN_BYTES
    ):
        raise LiteratureError("acquired Markdown is missing or exceeds the size limit")
    try:
        markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LiteratureError("acquired Markdown could not be read") from exc
    return payload, metadata_path, markdown


def _audit_passages(markdown: str, query_text: str) -> list[str]:
    terms = _normalized_words(query_text)
    scored = []
    for index, raw in enumerate(re.split(r"\n\s*\n", markdown)):
        passage = " ".join(raw.split())
        if not passage or passage == "---" or passage.startswith(("title:", "pmid:")):
            continue
        score = sum(term in passage.lower() for term in terms)
        scored.append((score, index, passage[:MAX_AUDIT_PASSAGE_CHARS]))
    relevant = [item for item in scored if item[0] > 0]
    selected = sorted(relevant or scored, key=lambda item: (-item[0], item[1]))[
        :MAX_AUDIT_PASSAGES_PER_ARTICLE
    ]
    return [item[2] for item in selected]


def build_acquired_article_audit(
    report: ScientificReport,
    retrieval: RetrievalEvidence,
) -> list[dict[str, Any]]:
    """Build a bounded, controller-read evidence packet for the Gemma audit."""

    claims_by_source: dict[str, list[dict[str, str]]] = {}
    for claim in report.claims:
        for source_id in claim.evidence_refs:
            destination = claims_by_source.setdefault(source_id, [])
            if len(destination) < 20:
                destination.append(
                    {"claim_id": claim.claim_id, "text": claim.text[:2_000]}
                )
    packet = []
    pubmed_sources = [source for source in report.sources if source.pmid]
    for source in pubmed_sources[:MAX_AUDIT_ARTICLES]:
        claims = claims_by_source.get(source.source_id, [])
        try:
            metadata, metadata_path, markdown = load_acquired_article_record(
                source, retrieval
            )
            query = " ".join(
                [source.supporting_passage, *(claim["text"] for claim in claims)]
            )
            article = metadata["article"]
            acquisition = metadata["acquisition"]
            bounded_metadata = {
                "article": {
                    key: article.get(key)
                    for key in (
                        "pmid",
                        "pmcid",
                        "doi",
                        "title",
                        "journal",
                        "year",
                        "canonical_url",
                        "publication_types",
                        "retracted",
                    )
                },
                "acquisition": {
                    key: acquisition.get(key)
                    for key in (
                        "citekey",
                        "status",
                        "markdown_status",
                        "pdf_status",
                        "pdf_path",
                        "markdown_path",
                        "license",
                        "rights_status",
                        "terms_warning",
                        "retracted",
                        "retrieved_at",
                        "pdf_verification",
                    )
                },
            }
            source_record = source.model_dump(mode="json")
            source_record["supporting_passage"] = source.supporting_passage[:2_000]
            packet.append(
                {
                    "source_id": source.source_id,
                    "source_record": source_record,
                    "acquisition_metadata": bounded_metadata,
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": sha256_file(metadata_path),
                    "markdown_sha256": sha256_file(
                        Path(metadata["acquisition"]["markdown_path"])
                    ),
                    "linked_claims": claims,
                    "controller_extracted_passages": _audit_passages(markdown, query),
                }
            )
        except (LiteratureError, OSError) as exc:
            source_record = source.model_dump(mode="json")
            source_record["supporting_passage"] = source.supporting_passage[:2_000]
            packet.append(
                {
                    "source_id": source.source_id,
                    "source_record": source_record,
                    "linked_claims": claims,
                    "controller_error": str(exc),
                    "controller_extracted_passages": [],
                }
            )
    if len(pubmed_sources) > MAX_AUDIT_ARTICLES:
        packet.append(
            {
                "controller_notice": (
                    f"Audit packet bounded to {MAX_AUDIT_ARTICLES} of "
                    f"{len(pubmed_sources)} PubMed sources."
                )
            }
        )
    return packet


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _first(node: ET.Element, *paths: str) -> str:
    for path in paths:
        value = _text(node.find(path))
        if value:
            return value
    return ""


def _year(article: ET.Element) -> str:
    candidates = [
        _first(article, ".//ArticleDate/Year"),
        _first(article, ".//JournalIssue/PubDate/Year"),
        _first(article, ".//PubDate/MedlineDate"),
    ]
    for candidate in candidates:
        match = re.search(r"(?:19|20)\d{2}", candidate)
        if match:
            return match.group(0)
    return "unknown-year"


def _parse_pubmed_article(article: ET.Element) -> dict[str, Any]:
    citation = article.find("MedlineCitation")
    pubmed_data = article.find("PubmedData")
    if citation is None:
        raise LiteratureError("PubMed returned a record without MedlineCitation")
    pmid = _first(citation, "PMID")
    if not _PMID.fullmatch(pmid):
        raise LiteratureError("PubMed returned an invalid PMID")
    article_node = citation.find("Article")
    if article_node is None:
        raise LiteratureError(f"PubMed record {pmid} has no Article metadata")
    identifiers: dict[str, str] = {}
    if pubmed_data is not None:
        # Only the record-level ArticleIdList is authoritative. PubmedData can
        # contain reference-list ArticleIds whose DOI/PMCID must never override
        # the identifiers of the requested PubMed article.
        for item in pubmed_data.findall("./ArticleIdList/ArticleId"):
            kind = (item.attrib.get("IdType") or "").lower()
            value = _text(item)
            if kind and value:
                identifiers[kind] = value
    authors: list[str] = []
    for author in article_node.findall(".//AuthorList/Author"):
        collective = _first(author, "CollectiveName")
        personal = " ".join(
            item
            for item in (_first(author, "ForeName"), _first(author, "LastName"))
            if item
        )
        name = collective or personal
        if name:
            authors.append(name)
    abstract_parts = [
        _text(item) for item in article_node.findall(".//Abstract/AbstractText")
    ]
    abstract = "\n\n".join(item for item in abstract_parts if item)
    publication_types = [
        _text(item)
        for item in article_node.findall(".//PublicationTypeList/PublicationType")
    ]
    retracted = any(
        "retracted publication" in item.lower() for item in publication_types if item
    )
    return {
        "pmid": pmid,
        "pmcid": identifiers.get("pmc"),
        "doi": identifiers.get("doi"),
        "title": _first(article_node, "ArticleTitle"),
        "journal": _first(
            article_node, ".//Journal/Title", ".//Journal/ISOAbbreviation"
        ),
        "year": _year(article_node),
        "authors": authors,
        "abstract": abstract,
        "publication_types": [item for item in publication_types if item],
        "retracted": retracted,
        "canonical_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def _parse_pubmed_xml(data: bytes) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise LiteratureError("PubMed returned malformed XML") from exc
    return [_parse_pubmed_article(item) for item in root.findall(".//PubmedArticle")]


def _slug(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    return (cleaned or fallback)[:48]


def _citekey(metadata: dict[str, Any]) -> str:
    first_author = str((metadata.get("authors") or ["article"])[0]).split()[-1]
    return (
        f"{_slug(first_author, 'article')}-{metadata.get('year') or 'undated'}-"
        f"pmid{metadata['pmid']}"
    )


def _normalized_words(value: str) -> set[str]:
    return {
        item
        for item in _WORD.findall(unicodedata.normalize("NFKD", value).lower())
        if len(item) >= 3
    }


def _verify_article_text(text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    compact = " ".join(text.split())
    if len(compact) < MIN_ARTICLE_TEXT_CHARS:
        raise LiteratureError("article text is too short to verify as full text")
    lower = compact.lower()
    identifiers = {
        "doi": str(metadata.get("doi") or "").lower(),
        "pmid": str(metadata.get("pmid") or "").lower(),
        "pmcid": str(metadata.get("pmcid") or "").lower(),
    }
    identifier_matches = [
        name for name, value in identifiers.items() if value and value in lower
    ]
    title_words = _normalized_words(str(metadata.get("title") or ""))
    text_words = _normalized_words(compact[:30_000])
    title_overlap = len(title_words & text_words) / max(1, len(title_words))
    title_matches = len(title_words) >= 3 and title_overlap >= 0.6
    if not identifier_matches and not title_matches:
        raise LiteratureError(
            "article text does not match the expected DOI/PMID/PMCID or title"
        )
    return {
        "identifier_matches": identifier_matches,
        "title_word_overlap": round(title_overlap, 3),
        "characters": len(compact),
    }


def _default_pdf_text(path: Path, pdftotext: Path) -> str:
    del path, pdftotext
    raise LiteratureError(
        "PDF parsing requires the authenticated no-network sandbox worker"
    )


@dataclass(frozen=True)
class RemotePdfTextExtractor:
    """Token-authenticated client for the isolated sandbox-worker PDF parser."""

    settings: SandboxSettings

    def __call__(self, path: Path, pdftotext: Path) -> str:
        del pdftotext
        if not self.settings.worker_url or not self.settings.worker_token:
            raise LiteratureError("sandbox-worker PDF extraction is not configured")
        request_id = str(uuid.uuid4())
        try:
            with httpx.Client(timeout=110, trust_env=False) as client:
                response = client.post(
                    f"{self.settings.worker_url}/extract-pdf-text",
                    headers={"Authorization": f"Bearer {self.settings.worker_token}"},
                    json={"request_id": request_id, "pdf_path": str(path.resolve())},
                )
        except httpx.HTTPError as exc:
            raise LiteratureError(
                "sandbox-worker PDF extraction request failed"
            ) from exc
        if response.is_error:
            try:
                detail = str(response.json().get("detail", "worker request failed"))
            except ValueError:
                detail = "worker returned a non-JSON error"
            raise LiteratureError(
                f"sandbox-worker PDF extraction failed ({response.status_code}): "
                f"{detail[:500]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LiteratureError("sandbox-worker returned invalid JSON") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_PDF_TEXT_BYTES:
            raise LiteratureError("sandbox-worker returned invalid PDF text")
        return text


@dataclass
class LiteratureAcquirer:
    workspace: Path
    settings: LiteratureSettings
    client: httpx.Client | None = None
    pdf_text_extractor: Callable[[Path, Path], str] = _default_pdf_text
    _last_ncbi_request: float = 0.0
    _request_lock: threading.Lock = field(default_factory=threading.Lock)
    _bibliography_lock: threading.Lock = field(
        default_factory=lambda: _BIBLIOGRAPHY_LOCK
    )

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        if not self.workspace.is_dir():
            raise LiteratureError(f"workspace does not exist: {self.workspace}")
        self._client_owned = self.client is None
        if self.client is None:
            self.client = httpx.Client(
                timeout=httpx.Timeout(45, connect=15),
                follow_redirects=False,
                trust_env=False,
                headers={
                    "User-Agent": "EvidenceBench/0.4 (PubMed literature acquisition)"
                },
            )

    def close(self) -> None:
        if self._client_owned and self.client is not None:
            self.client.close()

    def _require_ncbi_identity(self) -> None:
        if not self.settings.ncbi_email or "@" not in self.settings.ncbi_email:
            raise LiteratureError(
                "SCIENTIFIC_AGENT_NCBI_EMAIL must be configured for NCBI E-utilities"
            )
        if not self.settings.ncbi_tool:
            raise LiteratureError("SCIENTIFIC_AGENT_NCBI_TOOL must not be empty")

    def _rate_limit(self) -> None:
        # NCBI permits up to 3 requests/second without an API key.
        interval = 0.11 if self.settings.ncbi_api_key else 0.35
        with self._request_lock:
            remaining = interval - (time.monotonic() - self._last_ncbi_request)
            if remaining > 0:
                time.sleep(remaining)
            self._last_ncbi_request = time.monotonic()

    @staticmethod
    def _validate_literature_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise LiteratureError("literature request requires an HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise LiteratureError("literature request URL credentials are forbidden")
        host = parsed.hostname.lower()
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and (
            not address.is_global or address.is_multicast or address.is_unspecified
        ):
            raise LiteratureError(
                "literature request cannot target a private IP address"
            )
        if host not in ALLOWED_LITERATURE_HOSTS:
            raise LiteratureError("literature request host is not allow-listed")

    def _get(self, url: str, params: dict[str, Any], max_bytes: int) -> bytes:
        assert self.client is not None
        current_url = url
        current_params: dict[str, Any] | None = params
        for redirect_count in range(MAX_HTTP_REDIRECTS + 1):
            self._validate_literature_url(current_url)
            self._rate_limit()
            with self.client.stream(
                "GET",
                current_url,
                params=current_params,
                follow_redirects=False,
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise LiteratureError(
                            "literature redirect has no Location header"
                        )
                    if redirect_count >= MAX_HTTP_REDIRECTS:
                        raise LiteratureError("literature redirect limit exceeded")
                    current_url = urljoin(str(response.url), location)
                    current_params = None
                    continue
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise LiteratureError(
                            "remote article returned an invalid Content-Length"
                        ) from exc
                    if declared_size > max_bytes:
                        raise LiteratureError(
                            "remote article response exceeds the size limit"
                        )
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise LiteratureError(
                            "remote article response exceeds the size limit"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
        raise LiteratureError("literature redirect limit exceeded")

    def _eutils(self, endpoint: str, params: dict[str, Any]) -> bytes:
        self._require_ncbi_identity()
        fixed = {
            **params,
            "tool": self.settings.ncbi_tool,
            "email": self.settings.ncbi_email,
        }
        if self.settings.ncbi_api_key:
            fixed["api_key"] = self.settings.ncbi_api_key
        return self._get(f"{EUTILS_BASE}/{endpoint}", fixed, MAX_XML_BYTES)

    def _reference_dir(self, *parts: str) -> Path:
        current = self.workspace
        for part in ("references", *parts):
            if not part or part in {".", ".."} or "/" in part or "\\" in part:
                raise LiteratureError("invalid reference directory component")
            current = current / part
            if current.exists():
                info = current.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise LiteratureError(
                        f"reference path is not a regular directory: {current}"
                    )
            else:
                try:
                    current.mkdir(mode=0o700)
                except FileExistsError:
                    info = current.lstat()
                    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                        raise LiteratureError(
                            f"reference path is not a regular directory: {current}"
                        )
            resolved = current.resolve()
            if resolved != self.workspace and self.workspace not in resolved.parents:
                raise LiteratureError("reference path escaped the workspace")
            current = resolved
        return current

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(data)
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)

    def _fetch_metadata(self, pmid: str) -> dict[str, Any]:
        if not _PMID.fullmatch(pmid):
            raise LiteratureError(
                "PMID must contain 1 to 9 digits and cannot start with zero"
            )
        records = _parse_pubmed_xml(
            self._eutils(
                "efetch.fcgi",
                {"db": "pubmed", "id": pmid, "retmode": "xml"},
            )
        )
        if len(records) != 1 or records[0]["pmid"] != pmid:
            raise LiteratureError(f"PubMed did not return the requested PMID {pmid}")
        return records[0]

    def search(self, query: str, max_results: int = 10) -> dict[str, Any]:
        query = " ".join(query.split())
        if not query or len(query) > 500 or any(ord(char) < 32 for char in query):
            return {
                "error": "INVALID_QUERY",
                "reason": "query must be 1-500 printable characters",
            }
        if not 1 <= max_results <= MAX_SEARCH_RESULTS:
            return {
                "error": "INVALID_RESULT_LIMIT",
                "reason": f"max_results must be between 1 and {MAX_SEARCH_RESULTS}",
            }
        try:
            search_payload = json.loads(
                self._eutils(
                    "esearch.fcgi",
                    {
                        "db": "pubmed",
                        "term": query,
                        "retmode": "json",
                        "retmax": max_results,
                        "sort": "relevance",
                    },
                )
            )
            ids = [
                item
                for item in search_payload.get("esearchresult", {}).get("idlist", [])
                if _PMID.fullmatch(str(item))
            ]
            records: list[dict[str, Any]] = []
            if ids:
                records = _parse_pubmed_xml(
                    self._eutils(
                        "efetch.fcgi",
                        {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
                    )
                )
            by_pmid = {record["pmid"]: record for record in records}
            ordered = [by_pmid[item] for item in ids if item in by_pmid]
            payload = {
                "query": query,
                "retrieved_at": _utc_now(),
                "count": int(search_payload.get("esearchresult", {}).get("count", 0)),
                "articles": ordered,
                "notice": (
                    "PubMed metadata is supplied by NCBI; availability does not imply "
                    "endorsement, and article copyright/license terms still apply."
                ),
            }
            searches = self._reference_dir("searches")
            digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
            path = searches / f"pubmed-{digest}.json"
            self._atomic_write(
                path,
                (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
            )
            return {**payload, "artifacts": [str(path)]}
        except (
            LiteratureError,
            httpx.HTTPError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            return {"error": "PUBMED_SEARCH_FAILED", "reason": str(exc)}

    def _pmc_xml(self, metadata: dict[str, Any]) -> tuple[bytes | None, str | None]:
        pmcid = str(metadata.get("pmcid") or "")
        if not pmcid.upper().startswith("PMC"):
            return None, "No PMCID is linked to this PubMed record."
        try:
            data = self._eutils(
                "efetch.fcgi",
                {"db": "pmc", "id": pmcid[3:], "retmode": "xml"},
            )
            root = ET.fromstring(data)
            ids = {
                (item.attrib.get("pub-id-type") or "").lower(): _text(item)
                for item in root.findall(".//article-id")
            }
            expected = {
                "pmc": pmcid.upper(),
                "pmid": metadata["pmid"],
                "doi": str(metadata.get("doi") or "").lower(),
            }
            matches = []
            for key, value in expected.items():
                observed = str(ids.get(key, ""))
                if key == "pmc":
                    observed = observed.upper().removeprefix("PMC")
                    value = str(value).upper().removeprefix("PMC")
                if value and observed.lower() == str(value).lower():
                    matches.append(key)
            if not matches:
                raise LiteratureError(
                    "PMC XML did not contain the expected article identifier"
                )
            return data, None
        except (LiteratureError, httpx.HTTPError, ET.ParseError) as exc:
            return None, str(exc)

    def _oa_record(self, pmcid: str) -> tuple[dict[str, Any], str | None]:
        if not pmcid.upper().startswith("PMC"):
            return {}, "No PMCID is linked to this PubMed record."
        try:
            root = ET.fromstring(self._get(PMC_OA_URL, {"id": pmcid}, MAX_XML_BYTES))
            record = root.find(".//record")
            if record is None:
                error = _text(root.find(".//error")) or "PMC OA record is unavailable"
                return {}, error
            record_id = str(record.attrib.get("id") or "").upper()
            if record_id != pmcid.upper():
                return {}, "PMC OA-subset record did not match the requested PMCID"
            links = {
                (item.attrib.get("format") or "").lower(): item.attrib.get("href", "")
                for item in record.findall(".//link")
                if item.attrib.get("href")
            }
            return {
                "license": record.attrib.get("license") or "unknown",
                "retracted": (record.attrib.get("retracted") or "no").lower() == "yes",
                "citation": record.attrib.get("citation") or "",
                "links": links,
            }, None
        except (httpx.HTTPError, ET.ParseError, ValueError, LiteratureError) as exc:
            return {}, str(exc)

    @staticmethod
    def _oa_reuse_allowed(record: dict[str, Any]) -> tuple[bool, str]:
        license_value = " ".join(str(record.get("license") or "").upper().split())
        routes = [
            str(record.get("links", {}).get(kind) or "") for kind in ("tgz", "pdf")
        ]
        has_route = any(
            (
                value.startswith("ftp://ftp.ncbi.nlm.nih.gov/")
                or (
                    (parsed := urlparse(value)).scheme == "https"
                    and parsed.hostname in ALLOWED_DOWNLOAD_HOSTS
                    and not parsed.username
                    and not parsed.password
                )
            )
            for value in routes
            if value
        )
        reusable_license = (
            license_value == "CC0"
            or license_value.startswith("CC BY")
            or license_value in {"PUBLIC DOMAIN", "PDM"}
        )
        if not reusable_license:
            return False, "PMC OA-subset record has no explicit reusable license"
        if not has_route:
            return False, "PMC OA-subset record has no reusable full-text route"
        if record.get("retracted"):
            return False, "PMC OA-subset record is marked retracted"
        return True, "PMC OA-subset record and reusable license verified"

    def _download_oa_file(self, value: str, max_bytes: int) -> bytes:
        if value.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
            value = "https://ftp.ncbi.nlm.nih.gov/" + value.split("/", 3)[3]
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
            raise LiteratureError(
                "PMC OA record returned a non-allow-listed download URL"
            )
        return self._get(value, {}, max_bytes)

    @staticmethod
    def _nxml_from_archive(data: bytes) -> bytes:
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                candidates = []
                members = archive.getmembers()
                if len(members) > MAX_ARCHIVE_MEMBERS:
                    raise LiteratureError("PMC OA archive has too many members")
                for member in members:
                    path = Path(member.name)
                    if (
                        member.isfile()
                        and not member.issym()
                        and not member.islnk()
                        and not path.is_absolute()
                        and ".." not in path.parts
                        and path.suffix.lower() in {".nxml", ".xml"}
                    ):
                        candidates.append(member)
                if not candidates:
                    raise LiteratureError("PMC OA archive contains no safe NXML file")
                member = sorted(candidates, key=lambda item: item.size, reverse=True)[0]
                if member.size > MAX_XML_BYTES:
                    raise LiteratureError("PMC NXML exceeds the size limit")
                handle = archive.extractfile(member)
                if handle is None:
                    raise LiteratureError("PMC NXML could not be read")
                payload = handle.read(MAX_XML_BYTES + 1)
                if len(payload) > MAX_XML_BYTES:
                    raise LiteratureError("PMC NXML exceeds the size limit")
                return payload
        except tarfile.TarError as exc:
            raise LiteratureError("PMC OA archive is malformed") from exc

    @staticmethod
    def _pdfs_from_archive(data: bytes, max_pdf_bytes: int):
        """Yield bounded regular PDF members from a safe PMC OA archive."""

        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                members = archive.getmembers()
                if len(members) > MAX_ARCHIVE_MEMBERS:
                    raise LiteratureError("PMC OA archive has too many members")
                candidates = []
                oversized = False
                for member in members:
                    path = Path(member.name)
                    if (
                        not member.isfile()
                        or member.issym()
                        or member.islnk()
                        or path.is_absolute()
                        or ".." in path.parts
                        or path.suffix.lower() != ".pdf"
                    ):
                        continue
                    if member.size > max_pdf_bytes:
                        oversized = True
                        continue
                    candidates.append(member)
                if not candidates:
                    if oversized:
                        raise LiteratureError(
                            "PMC OA archive PDF exceeds the configured size limit"
                        )
                    raise LiteratureError("PMC OA archive contains no safe PDF file")
                candidates.sort(key=lambda item: item.size, reverse=True)
                for member in candidates[:MAX_ARCHIVE_PDF_CANDIDATES]:
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    payload = handle.read(max_pdf_bytes + 1)
                    if len(payload) > max_pdf_bytes:
                        raise LiteratureError(
                            "PMC OA archive PDF exceeds the configured size limit"
                        )
                    yield member.name, payload
        except tarfile.TarError as exc:
            raise LiteratureError("PMC OA archive is malformed") from exc

    @staticmethod
    def _jats_markdown(data: bytes, metadata: dict[str, Any], status: str) -> str:
        try:
            root = ET.fromstring(data)
        except ET.ParseError as exc:
            raise LiteratureError("article NXML is malformed") from exc
        title = _first(root, ".//article-title") or metadata["title"]
        lines = [
            "---",
            f"title: {json.dumps(title, ensure_ascii=False)}",
            f"pmid: {json.dumps(metadata['pmid'])}",
            f"pmcid: {json.dumps(metadata.get('pmcid'))}",
            f"doi: {json.dumps(metadata.get('doi'))}",
            f"acquisition_status: {json.dumps(status)}",
            "---",
            "",
            f"# {title}",
            "",
        ]
        abstract = _first(root, ".//abstract")
        if abstract:
            lines.extend(["## Abstract", "", abstract, ""])
        body = root.find(".//body")
        if body is not None:
            body_added = False
            for paragraph in body.findall("./p"):
                value = _text(paragraph)
                if value:
                    lines.extend([value, ""])
                    body_added = True
            for section in body.findall(".//sec"):
                heading = _first(section, "title")
                if heading:
                    lines.extend([f"## {heading}", ""])
                for paragraph in section.findall("./p"):
                    value = _text(paragraph)
                    if value:
                        lines.extend([value, ""])
                        body_added = True
            for table in body.findall(".//table-wrap"):
                value = _text(table)
                if value:
                    lines.extend(["### Table text", "", value, ""])
                    body_added = True
            if not body_added:
                body_text = _text(body)
                if body_text:
                    lines.extend(["## Full text", "", body_text, ""])
        markdown = "\n".join(lines).strip() + "\n"
        _verify_article_text(markdown, metadata)
        return markdown

    def _pubtator_markdown(
        self, metadata: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        try:
            raw = self._get(
                PUBTATOR_URL,
                {"pmids": metadata["pmid"], "full": "true"},
                MAX_XML_BYTES,
            )
            decoded = raw.decode("utf-8")
            try:
                payload: Any = json.loads(decoded)
            except json.JSONDecodeError:
                payload = [
                    json.loads(line) for line in decoded.splitlines() if line.strip()
                ]
            documents = payload if isinstance(payload, list) else [payload]
            passages: list[str] = []
            for document in documents:
                if not isinstance(document, dict):
                    continue
                for passage in document.get("passages", []):
                    value = passage.get("text") if isinstance(passage, dict) else None
                    if isinstance(value, str) and value.strip():
                        passages.append(value.strip())
            text = "\n\n".join(passages)
            verification = _verify_article_text(text, metadata)
            markdown = (
                f"---\ntitle: {json.dumps(metadata['title'], ensure_ascii=False)}\n"
                f"pmid: {json.dumps(metadata['pmid'])}\n"
                f"pmcid: {json.dumps(metadata.get('pmcid'))}\n"
                f"doi: {json.dumps(metadata.get('doi'))}\n"
                'acquisition_status: "pubtator_full_text"\n---\n\n'
                f"# {metadata['title']}\n\n{text}\n"
            )
            return markdown, json.dumps(verification, sort_keys=True)
        except (
            LiteratureError,
            httpx.HTTPError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            return None, str(exc)

    @staticmethod
    def _abstract_markdown(metadata: dict[str, Any]) -> str:
        abstract = str(metadata.get("abstract") or "").strip()
        return (
            f"---\ntitle: {json.dumps(metadata['title'], ensure_ascii=False)}\n"
            f"pmid: {json.dumps(metadata['pmid'])}\n"
            f"pmcid: {json.dumps(metadata.get('pmcid'))}\n"
            f"doi: {json.dumps(metadata.get('doi'))}\n"
            'acquisition_status: "abstract_only"\n---\n\n'
            f"# {metadata['title']}\n\n## Abstract\n\n"
            f"{abstract or 'No abstract was supplied by PubMed.'}\n"
        )

    def _verify_pdf(
        self, path: Path, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        size = path.stat().st_size
        if size < MIN_PDF_BYTES:
            raise LiteratureError(f"PDF is too small ({size} bytes)")
        if size > self.settings.max_pdf_bytes:
            raise LiteratureError("PDF exceeds the configured size limit")
        with path.open("rb") as handle:
            signature = handle.read(1024)
        if b"%PDF-" not in signature:
            raise LiteratureError("downloaded file does not have a PDF signature")
        text = self.pdf_text_extractor(path, self.settings.pdftotext)
        verification = _verify_article_text(text, metadata)
        return text, {"bytes": size, "sha256": sha256_file(path), **verification}

    def _write_bibliography(
        self,
        citekey: str,
        metadata: dict[str, Any],
        record: dict[str, Any],
    ) -> Path:
        path = self._reference_dir() / "bibliography.md"
        header = (
            "# Workspace bibliography\n\n"
            "> Metadata and availability are provided through NCBI services. NCBI "
            "does not endorse this analysis; copyright and license terms remain "
            "with each article.\n\n"
            "| Citekey | Citation | Identifiers | PDF | Markdown | Access status "
            "| License | Rights status | Retraction |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        authors = metadata.get("authors") or []
        author_text = f"{', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}"
        citation = (
            f"{author_text}. {metadata['title']}. {metadata.get('journal') or ''} "
            f"({metadata.get('year') or 'n.d.'})."
        )
        identifiers = [f"PMID {metadata['pmid']}"]
        if metadata.get("doi"):
            identifiers.append(f"DOI {metadata['doi']}")
        if metadata.get("pmcid"):
            identifiers.append(str(metadata["pmcid"]))
        pdf = f"[PDF](pdfs/{citekey}.pdf)" if record.get("pdf_path") else "Unavailable"
        markdown = f"[Markdown](markdown/{citekey}.md)"
        values = [
            citekey,
            citation,
            "; ".join(identifiers),
            pdf,
            markdown,
            str(record["status"]),
            str(record["license"]),
            str(record["rights_status"]),
            "Retracted" if record["retracted"] else "No retraction flag returned",
        ]
        row = (
            "| "
            + " | ".join(
                value.replace("|", "\\|").replace("\n", " ") for value in values
            )
            + " |"
        )
        with self._bibliography_lock:
            existing = path.read_text(encoding="utf-8") if path.is_file() else header
            rows = [
                line
                for line in existing.splitlines()
                if not line.startswith(f"| {citekey} |")
            ]
            rows.append(row)
            self._atomic_write(path, ("\n".join(rows).rstrip() + "\n").encode("utf-8"))
        return path

    def _finalize(
        self,
        metadata: dict[str, Any],
        markdown: str,
        *,
        status: str,
        markdown_status: str,
        pdf_temporary: Path | None,
        pdf_verification: dict[str, Any] | None,
        oa_record: dict[str, Any],
        acquisition_notes: list[str],
        rights_status: str,
        terms_warning: str,
    ) -> dict[str, Any]:
        citekey = _citekey(metadata)
        pdf_path: Path | None = None
        if pdf_temporary is not None:
            pdf_path = self._reference_dir("pdfs") / f"{citekey}.pdf"
            os.replace(pdf_temporary, pdf_path)
            pdf_path.chmod(0o600)
        markdown_path = self._reference_dir("markdown") / f"{citekey}.md"
        record = {
            "citekey": citekey,
            "status": status,
            "markdown_status": markdown_status,
            "pdf_status": "verified" if pdf_path else "unavailable",
            "pdf_path": str(pdf_path) if pdf_path else None,
            "markdown_path": str(markdown_path),
            "pdf_verification": pdf_verification,
            "license": oa_record.get("license") or "unknown",
            "retracted": bool(
                oa_record.get("retracted", False) or metadata.get("retracted", False)
            ),
            "acquisition_notes": acquisition_notes,
            "retrieved_at": _utc_now(),
            "rights_status": rights_status,
            "terms_warning": terms_warning,
        }
        markdown_lines = markdown.splitlines()
        if markdown_lines and markdown_lines[0] == "---":
            try:
                end = markdown_lines.index("---", 1)
                markdown_lines[end:end] = [
                    f"license: {json.dumps(record['license'], ensure_ascii=False)}",
                    f"retracted: {json.dumps(record['retracted'])}",
                    f"retrieved_at: {json.dumps(record['retrieved_at'])}",
                    f"rights_status: {json.dumps(record['rights_status'])}",
                    f"terms_warning: {json.dumps(record['terms_warning'], ensure_ascii=False)}",
                ]
                markdown = "\n".join(markdown_lines).rstrip() + "\n"
            except ValueError:
                pass
        self._atomic_write(markdown_path, markdown.encode("utf-8"))
        record["markdown_path"] = str(markdown_path)
        metadata_path = self._reference_dir("metadata") / f"{citekey}.json"
        self._atomic_write(
            metadata_path,
            (
                json.dumps(
                    {"article": metadata, "acquisition": record},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        bibliography = self._write_bibliography(citekey, metadata, record)
        artifacts = [str(metadata_path), str(markdown_path), str(bibliography)]
        if pdf_path:
            artifacts.append(str(pdf_path))
        publication_types = {
            str(item).lower() for item in metadata.get("publication_types", [])
        }
        source_type = (
            "review"
            if any("review" in item for item in publication_types)
            else "primary_study"
        )
        return {
            "status": status,
            "article": metadata,
            "acquisition": record,
            "source_record": {
                "title": metadata["title"],
                "url": metadata["canonical_url"],
                "source_type": source_type,
                "retrieved_at": record["retrieved_at"],
                "pmid": metadata["pmid"],
                "pmcid": metadata.get("pmcid"),
                "doi": metadata.get("doi"),
                "citekey": citekey,
                "license": record["license"],
                "retracted": record["retracted"],
                "rights_status": record["rights_status"],
                "terms_warning": record["terms_warning"],
                "local_pdf_path": str(pdf_path) if pdf_path else None,
                "local_markdown_path": str(markdown_path),
                "full_text_status": status,
            },
            "artifacts": artifacts,
            "notice": (
                "NCBI does not endorse downstream analyses. Preserve the recorded "
                "license/copyright terms; a missing PDF is not evidence of absence."
            ),
        }

    def acquire(self, pmid: str) -> dict[str, Any]:
        temporary_pdf: Path | None = None
        try:
            metadata = self._fetch_metadata(pmid)
            oa_record, oa_error = self._oa_record(str(metadata.get("pmcid") or ""))
            notes = [item for item in [oa_error] if item]
            oa_allowed, oa_reason = self._oa_reuse_allowed(oa_record)
            if not oa_allowed:
                notes.append(f"Automatic full text unavailable: {oa_reason}.")
                markdown = self._abstract_markdown(metadata)
                return self._finalize(
                    metadata,
                    markdown,
                    status="abstract_only",
                    markdown_status="abstract_only",
                    pdf_temporary=None,
                    pdf_verification=None,
                    oa_record=oa_record,
                    acquisition_notes=notes,
                    rights_status="metadata_abstract_only_no_reuse_rights",
                    terms_warning=(
                        "No reusable PMC OA-subset license and route were verified; "
                        "only PubMed metadata and abstract text were stored."
                    ),
                )

            notes.append(oa_reason)
            nxml, xml_error = self._pmc_xml(metadata)
            if xml_error:
                notes.append(xml_error)
            markdown: str | None = None
            markdown_status = "unavailable"
            archive_data: bytes | None = None

            def oa_archive() -> bytes:
                nonlocal archive_data
                if archive_data is not None:
                    return archive_data
                archive_link = oa_record.get("links", {}).get("tgz")
                if not archive_link:
                    raise LiteratureError("PMC OA record has no archive link")
                archive_data = self._download_oa_file(
                    archive_link, self.settings.max_archive_bytes
                )
                return archive_data

            if nxml is not None:
                markdown = self._jats_markdown(nxml, metadata, "pmc_full_text")
                markdown_status = "pmc_full_text"
            elif oa_record.get("links", {}).get("tgz"):
                try:
                    nxml = self._nxml_from_archive(oa_archive())
                    markdown = self._jats_markdown(nxml, metadata, "pmc_oa_archive")
                    markdown_status = "pmc_oa_archive"
                except LiteratureError as exc:
                    notes.append(str(exc))
            if markdown is None:
                markdown, pubtator_note = self._pubtator_markdown(metadata)
                if markdown is not None:
                    markdown_status = "pubtator_full_text"
                elif pubtator_note:
                    notes.append(pubtator_note)
            pdf_verification = None
            pdf_link = oa_record.get("links", {}).get("pdf")
            if pdf_link:
                try:
                    data = self._download_oa_file(pdf_link, self.settings.max_pdf_bytes)
                    temporary_pdf = (
                        self._reference_dir("pdfs")
                        / f".download-{uuid.uuid4().hex}.pdf"
                    )
                    self._atomic_write(temporary_pdf, data)
                    _, pdf_verification = self._verify_pdf(temporary_pdf, metadata)
                except (LiteratureError, httpx.HTTPError) as exc:
                    notes.append(f"PDF unavailable: {exc}")
                    if temporary_pdf is not None:
                        temporary_pdf.unlink(missing_ok=True)
                        temporary_pdf = None
            if temporary_pdf is None and oa_record.get("links", {}).get("tgz"):
                archive_errors = []
                try:
                    for member_name, data in self._pdfs_from_archive(
                        oa_archive(), self.settings.max_pdf_bytes
                    ):
                        candidate = (
                            self._reference_dir("pdfs")
                            / f".archive-{uuid.uuid4().hex}.pdf"
                        )
                        try:
                            self._atomic_write(candidate, data)
                            _, pdf_verification = self._verify_pdf(candidate, metadata)
                            temporary_pdf = candidate
                            notes.append(
                                f"PDF acquired from verified PMC OA archive member: "
                                f"{member_name}"
                            )
                            break
                        except LiteratureError as exc:
                            archive_errors.append(f"{member_name}: {exc}")
                            candidate.unlink(missing_ok=True)
                    if temporary_pdf is None and archive_errors:
                        raise LiteratureError("; ".join(archive_errors[:3]))
                except (LiteratureError, httpx.HTTPError) as exc:
                    notes.append(f"Archive PDF unavailable: {exc}")
            if markdown is None:
                markdown = self._abstract_markdown(metadata)
                markdown_status = "abstract_only"
            has_full_text = markdown_status != "abstract_only"
            status = (
                "full_text_with_pdf"
                if has_full_text and temporary_pdf is not None
                else "full_text_markdown_only"
                if has_full_text
                else "abstract_only"
            )
            return self._finalize(
                metadata,
                markdown,
                status=status,
                markdown_status=markdown_status,
                pdf_temporary=temporary_pdf,
                pdf_verification=pdf_verification,
                oa_record=oa_record,
                acquisition_notes=notes,
                rights_status="pmc_oa_reuse_allowed",
                terms_warning=AUTOMATIC_OA_TERMS_WARNING,
            )
        except (LiteratureError, httpx.HTTPError, ValueError, OSError) as exc:
            if temporary_pdf is not None:
                temporary_pdf.unlink(missing_ok=True)
            return {
                "error": "ARTICLE_ACQUISITION_FAILED",
                "reason": str(exc),
                "pmid": pmid,
            }

    def import_browser_pdf(self, pmid: str, filename: str) -> dict[str, Any]:
        temporary: Path | None = None
        try:
            if (
                not _SAFE_FILENAME.fullmatch(filename)
                or Path(filename).name != filename
            ):
                raise LiteratureError("filename must be one plain .pdf basename")
            inbox = self.settings.browser_downloads_dir
            if not inbox.is_dir() or inbox.is_symlink():
                raise LiteratureError("managed browser download inbox is unavailable")
            candidate = inbox / filename
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise LiteratureError(
                    "browser download must be a regular non-symlink file"
                )
            if candidate.resolve().parent != inbox.resolve():
                raise LiteratureError("browser download escaped the managed inbox")
            if info.st_size > self.settings.max_pdf_bytes:
                raise LiteratureError("browser PDF exceeds the configured size limit")
            metadata = self._fetch_metadata(pmid)
            temporary = self._reference_dir("pdfs") / f".manual-{uuid.uuid4().hex}.pdf"
            with candidate.open("rb") as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            temporary.chmod(0o600)
            text, verification = self._verify_pdf(temporary, metadata)
            markdown = (
                f"---\ntitle: {json.dumps(metadata['title'], ensure_ascii=False)}\n"
                f"pmid: {json.dumps(metadata['pmid'])}\n"
                f"pmcid: {json.dumps(metadata.get('pmcid'))}\n"
                f"doi: {json.dumps(metadata.get('doi'))}\n"
                'acquisition_status: "verified_manual_browser_pdf"\n---\n\n'
                f"# {metadata['title']}\n\n{text.strip()}\n"
            )
            oa_record, oa_error = self._oa_record(str(metadata.get("pmcid") or ""))
            notes = ["Imported from the managed Evidence Bench browser download inbox."]
            if oa_error:
                notes.append(oa_error)
            return self._finalize(
                metadata,
                markdown,
                status="verified_manual_browser_pdf",
                markdown_status="pdf_text",
                pdf_temporary=temporary,
                pdf_verification=verification,
                oa_record=oa_record,
                acquisition_notes=notes,
                rights_status="private_user_provided",
                terms_warning=MANUAL_PDF_TERMS_WARNING,
            )
        except (LiteratureError, httpx.HTTPError, OSError) as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            return {
                "error": "BROWSER_PDF_IMPORT_FAILED",
                "reason": str(exc),
                "pmid": pmid,
            }

    def list_browser_downloads(self) -> dict[str, Any]:
        """List only direct regular PDFs in the service-owned browser inbox."""

        inbox = self.settings.browser_downloads_dir
        try:
            if not inbox.is_dir() or inbox.is_symlink():
                raise LiteratureError("managed browser download inbox is unavailable")
            files = []
            for path in sorted(inbox.iterdir(), key=lambda item: item.name.lower()):
                if not _SAFE_FILENAME.fullmatch(path.name):
                    continue
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    continue
                if info.st_size > self.settings.max_pdf_bytes:
                    continue
                files.append(
                    {
                        "filename": path.name,
                        "bytes": info.st_size,
                        "modified_at": datetime.fromtimestamp(
                            info.st_mtime, UTC
                        ).isoformat(),
                    }
                )
                if len(files) >= 100:
                    break
            return {"status": "available", "pdfs": files}
        except (LiteratureError, OSError) as exc:
            return {"error": "BROWSER_DOWNLOAD_INBOX_UNAVAILABLE", "reason": str(exc)}

    def search_acquired_article(
        self,
        citekey: str,
        query: str,
        max_matches: int = 10,
    ) -> dict[str, Any]:
        """Return bounded matching passages from one acquired Markdown article."""

        query = " ".join(query.split())
        if not _CITEKEY.fullmatch(citekey):
            return {
                "error": "INVALID_CITEKEY",
                "reason": "citekey has an invalid format",
            }
        if not query or len(query) > 300 or not query.isprintable():
            return {
                "error": "INVALID_QUERY",
                "reason": "query must be 1-300 printable characters",
            }
        if not 1 <= max_matches <= 20:
            return {
                "error": "INVALID_MATCH_LIMIT",
                "reason": "max_matches must be 1-20",
            }
        try:
            path = self._reference_dir("markdown") / f"{citekey}.md"
            if not path.is_file() or path.is_symlink():
                raise LiteratureError("acquired article Markdown was not found")
            data = path.read_bytes()
            if len(data) > MAX_XML_BYTES:
                raise LiteratureError(
                    "acquired article Markdown exceeds the search limit"
                )
            text = data.decode("utf-8", errors="replace")
            terms = _normalized_words(query)
            if not terms:
                terms = {query.lower()}
            matches: list[dict[str, Any]] = []
            for index, passage in enumerate(re.split(r"\n\s*\n", text), 1):
                compact = " ".join(passage.split())
                if not compact:
                    continue
                lower = compact.lower()
                score = sum(term in lower for term in terms)
                if score == 0:
                    continue
                matches.append(
                    {
                        "passage": index,
                        "matched_terms": score,
                        "text": compact[:2_000],
                    }
                )
            matches.sort(key=lambda item: (-item["matched_terms"], item["passage"]))
            return {
                "status": "searched",
                "citekey": citekey,
                "query": query,
                "matches": matches[:max_matches],
                "truncated": len(matches) > max_matches,
                "article_path": str(path),
            }
        except (LiteratureError, OSError) as exc:
            return {"error": "ARTICLE_SEARCH_FAILED", "reason": str(exc)}


def build_literature_tools(acquirer: LiteratureAcquirer):
    def list_browser_downloads() -> dict[str, Any]:
        """List PDF basenames in the managed Evidence Bench browser inbox."""

        return acquirer.list_browser_downloads()

    def search_acquired_article(
        citekey: str,
        query: str,
        max_matches: int = 10,
    ) -> dict[str, Any]:
        """Search bounded passages in one locally acquired article Markdown file.

        Args:
            citekey: Exact citekey returned by acquire_pubmed_article.
            query: Literal concepts or phrase to locate in the article.
            max_matches: Maximum number of matching passages, from 1 through 20.
        """

        return acquirer.search_acquired_article(citekey, query, max_matches)

    def search_pubmed(query: str, max_results: int = 10) -> dict[str, Any]:
        """Search PubMed through NCBI E-utilities and return verified metadata.

        Args:
            query: A PubMed search expression (maximum 500 characters).
            max_results: Number of metadata records to return, from 1 through 20.
        """

        return acquirer.search(query, max_results)

    def acquire_pubmed_article(pmid: str) -> dict[str, Any]:
        """Acquire one PubMed article using legitimate open-access routes.

        Stores metadata and searchable Markdown in workspace references. A PDF is
        stored only when its signature, size, and article identity verify. Missing
        full text or PDF remains an explicit acquisition status.

        Args:
            pmid: The exact numeric PubMed identifier selected from search results.
        """

        return acquirer.acquire(pmid)

    def import_browser_downloaded_pdf(pmid: str, filename: str) -> dict[str, Any]:
        """Verify and import one PDF from the managed Evidence Bench browser inbox.

        The filename must be a plain basename in /browser-downloads. This tool
        cannot navigate arbitrary filesystem paths and never removes the inbox copy.

        Args:
            pmid: Exact numeric PubMed identifier for content verification.
            filename: Plain PDF filename visible in the managed browser inbox.
        """

        return acquirer.import_browser_pdf(pmid, filename)

    return [
        search_pubmed,
        acquire_pubmed_article,
        search_acquired_article,
        list_browser_downloads,
        import_browser_downloaded_pdf,
    ]
