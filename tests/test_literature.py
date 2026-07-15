import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from scientific_agent.config import LiteratureSettings, SandboxSettings, Settings
from scientific_agent.linting import validate_report
from scientific_agent.literature import (
    LiteratureAcquirer,
    LiteratureError,
    RemotePdfTextExtractor,
    _default_pdf_text,
    _parse_pubmed_xml,
    build_acquired_article_audit,
    load_acquired_article_record,
)
from scientific_agent.policy import ToolPolicy, default_allowed_tools
from scientific_agent.provenance import EventLedger
from scientific_agent.reporting import materialize_references, render_report_markdown
from scientific_agent.schemas import (
    CheckSpec,
    ClaimRecord,
    ComputationEvidence,
    DeterministicValidation,
    EvidenceStatus,
    MasterPlan,
    PlanningResult,
    PlanProposal,
    PlanStep,
    RetrievalEvidence,
    ScientificReport,
    SourceRecord,
    TaskSpec,
    VerificationReport,
)


PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID>12345678</PMID><Article>
<ArticleTitle>A verified treatment effect in a controlled study</ArticleTitle>
<Journal><Title>Journal of Reproducible Results</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
<AuthorList><Author><ForeName>Ada</ForeName><LastName>Lovelace</LastName></Author></AuthorList>
<Abstract><AbstractText>We report a verified effect with reproducible methods and transparent uncertainty.</AbstractText></Abstract>
<PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
</Article></MedlineCitation><PubmedData><ArticleIdList>
<ArticleId IdType="pubmed">12345678</ArticleId>
<ArticleId IdType="pmc">PMC9876543</ArticleId>
<ArticleId IdType="doi">10.1000/verified.2024.1</ArticleId>
</ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""

JATS_XML = (
    """<?xml version="1.0"?><article><front><article-meta>
<article-id pub-id-type="pmid">12345678</article-id>
<article-id pub-id-type="pmc">PMC9876543</article-id>
<article-id pub-id-type="doi">10.1000/verified.2024.1</article-id>
<title-group><article-title>A verified treatment effect in a controlled study</article-title></title-group>
<abstract><p>Reproducible evidence was evaluated.</p></abstract>
</article-meta></front><body><sec><title>Introduction</title><p>"""
    + "A verified treatment effect was evaluated with reproducible methods. " * 16
    + "</p></sec><sec><title>Results</title><p>"
    + "The controlled study reports uncertainty and complete analysis details. " * 16
    + "</p></sec></body></article>"
).encode()


def _settings(downloads: Path) -> LiteratureSettings:
    return LiteratureSettings(
        ncbi_email="maintainer@example.org",
        ncbi_tool="evidence_bench_tests",
        ncbi_api_key="test-key",
        browser_downloads_dir=downloads,
        pdftotext=Path("/usr/bin/pdftotext"),
        max_pdf_bytes=2 * 1024 * 1024,
        max_archive_bytes=2 * 1024 * 1024,
    )


def _client(*, include_pdf: bool = False) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = request.url.params
        if path.endswith("/esearch.fcgi"):
            return httpx.Response(
                200,
                json={"esearchresult": {"count": "1", "idlist": ["12345678"]}},
            )
        if path.endswith("/efetch.fcgi") and params.get("db") == "pubmed":
            assert params.get("tool") == "evidence_bench_tests"
            assert params.get("email") == "maintainer@example.org"
            return httpx.Response(200, content=PUBMED_XML)
        if path.endswith("/efetch.fcgi") and params.get("db") == "pmc":
            return httpx.Response(200, content=JATS_XML)
        if path.endswith("/oa.fcgi"):
            pdf_link = (
                '<link format="pdf" href="https://ftp.ncbi.nlm.nih.gov/pub/fake/article.pdf" />'
                if include_pdf
                else ""
            )
            return httpx.Response(
                200,
                content=(
                    '<OA><records><record id="PMC9876543" license="CC BY" retracted="no">'
                    f'{pdf_link}<link format="tgz" '
                    'href="https://ftp.ncbi.nlm.nih.gov/pub/fake/article.tar.gz" />'
                    "</record></records></OA>"
                ).encode(),
            )
        if path.endswith("/article.pdf"):
            return httpx.Response(200, content=b"%PDF-1.7\n" + b"x" * 12_000)
        if path.endswith("/article.tar.gz"):
            member = tarfile.TarInfo("article/article.nxml")
            return httpx.Response(200, content=_tar_gz([(member, JATS_XML)]))
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def _pdf_text(_path: Path, _pdftotext: Path) -> str:
    return (
        "A verified treatment effect in a controlled study. "
        "DOI 10.1000/verified.2024.1. "
        + "Complete methods and results establish searchable article text. "
        * 20
    )


def test_local_pdf_parser_fails_closed_and_remote_parser_uses_worker_token(
    tmp_path: Path,
    monkeypatch,
):
    pdf = tmp_path / "article.pdf"
    pdf.write_bytes(b"%PDF-1.7\nfixture")
    with pytest.raises(LiteratureError, match="sandbox worker"):
        _default_pdf_text(pdf, Path("/usr/bin/pdftotext"))

    observed = {}

    class Response:
        is_error = False

        @staticmethod
        def json():
            return {"text": "verified article text"}

    class Client:
        def __init__(self, **kwargs):
            observed["client"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def post(url, *, headers, json):
            observed.update({"url": url, "headers": headers, "json": json})
            return Response()

    monkeypatch.setattr("scientific_agent.literature.httpx.Client", Client)
    extractor = RemotePdfTextExtractor(
        SandboxSettings(worker_url="http://sandbox-worker:8090", worker_token="s" * 32)
    )

    assert extractor(pdf, Path("/ignored")) == "verified article text"
    assert observed["client"]["trust_env"] is False
    assert observed["headers"]["Authorization"] == f"Bearer {'s' * 32}"
    assert observed["url"].endswith("/extract-pdf-text")
    assert observed["json"]["pdf_path"] == str(pdf.resolve())


def test_pubmed_search_and_full_text_markdown_are_persisted(tmp_path: Path):
    workspace = tmp_path / "workspace"
    downloads = tmp_path / "downloads"
    workspace.mkdir()
    downloads.mkdir()
    acquirer = LiteratureAcquirer(
        workspace,
        _settings(downloads),
        client=_client(),
        pdf_text_extractor=_pdf_text,
    )

    search = acquirer.search("controlled treatment effect", max_results=5)
    acquired = acquirer.acquire("12345678")
    passages = acquirer.search_acquired_article(
        acquired["acquisition"]["citekey"], "uncertainty analysis"
    )

    assert search["articles"][0]["doi"] == "10.1000/verified.2024.1"
    assert Path(search["artifacts"][0]).is_file()
    assert acquired["status"] == "full_text_markdown_only"
    assert acquired["article"]["pmcid"] == "PMC9876543"
    assert acquired["acquisition"]["license"] == "CC BY"
    assert acquired["acquisition"]["rights_status"] == "pmc_oa_reuse_allowed"
    markdown = Path(acquired["acquisition"]["markdown_path"])
    assert markdown.is_file()
    assert "## Results" in markdown.read_text(encoding="utf-8")
    assert acquired["acquisition"]["pdf_path"] is None
    assert passages["matches"]
    assert "uncertainty" in passages["matches"][0]["text"].lower()
    bibliography = workspace / "references" / "bibliography.md"
    assert "Unavailable" in bibliography.read_text(encoding="utf-8")
    assert all(workspace in Path(path).parents for path in acquired["artifacts"])


def test_ncbi_contact_identity_is_required(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = LiteratureSettings(
        ncbi_email="",
        ncbi_tool="evidence_bench_tests",
        browser_downloads_dir=tmp_path / "downloads",
    )
    acquirer = LiteratureAcquirer(workspace, settings, client=_client())

    result = acquirer.search("treatment effect")

    assert result["error"] == "PUBMED_SEARCH_FAILED"
    assert "NCBI_EMAIL" in result["reason"]


@pytest.mark.parametrize(
    "location",
    ["https://evil.example/article", "https://127.0.0.1/private"],
)
def test_ncbi_redirect_to_untrusted_or_private_host_is_rejected(
    tmp_path: Path,
    location: str,
):
    workspace = tmp_path / "workspace"
    downloads = tmp_path / "downloads"
    workspace.mkdir()
    downloads.mkdir()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": location})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )
    acquirer = LiteratureAcquirer(workspace, _settings(downloads), client=client)

    result = acquirer.search("treatment effect")

    assert result["error"] == "PUBMED_SEARCH_FAILED"
    assert "allow-listed" in result["reason"] or "private IP" in result["reason"]


def test_ncbi_redirect_chain_is_bounded(tmp_path: Path):
    workspace = tmp_path / "workspace"
    downloads = tmp_path / "downloads"
    workspace.mkdir()
    downloads.mkdir()

    def handler(request: httpx.Request) -> httpx.Response:
        hop = int(request.url.params.get("hop", "0"))
        return httpx.Response(
            302,
            headers={
                "Location": (
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
                    f"esearch.fcgi?hop={hop + 1}"
                )
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    acquirer = LiteratureAcquirer(workspace, _settings(downloads), client=client)

    result = acquirer.search("treatment effect")

    assert result["error"] == "PUBMED_SEARCH_FAILED"
    assert "redirect limit" in result["reason"]


def test_default_ncbi_client_ignores_proxy_environment(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    downloads = tmp_path / "downloads"
    workspace.mkdir()
    downloads.mkdir()
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

    acquirer = LiteratureAcquirer(workspace, _settings(downloads))
    try:
        assert acquirer.client is not None
        assert acquirer.client._trust_env is False
        assert acquirer.client.follow_redirects is False
    finally:
        acquirer.close()


def test_open_access_pdf_requires_signature_size_and_article_identity(tmp_path: Path):
    workspace = tmp_path / "workspace"
    downloads = tmp_path / "downloads"
    workspace.mkdir()
    downloads.mkdir()
    acquirer = LiteratureAcquirer(
        workspace,
        _settings(downloads),
        client=_client(include_pdf=True),
        pdf_text_extractor=_pdf_text,
    )

    acquired = acquirer.acquire("12345678")

    assert acquired["status"] == "full_text_with_pdf"
    pdf = Path(acquired["acquisition"]["pdf_path"])
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert acquired["acquisition"]["pdf_verification"]["identifier_matches"] == ["doi"]
    assert acquired["source_record"]["local_pdf_path"] == str(pdf)


def test_pmc_without_reusable_oa_record_stores_only_pubmed_abstract(tmp_path: Path):
    workspace = tmp_path / "workspace"
    downloads = tmp_path / "downloads"
    workspace.mkdir()
    downloads.mkdir()
    protected_routes_called: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = request.url.params
        if path.endswith("/efetch.fcgi") and params.get("db") == "pubmed":
            return httpx.Response(200, content=PUBMED_XML)
        if path.endswith("/oa.fcgi"):
            return httpx.Response(
                200,
                content=b'<OA><error code="idIsNotOpenAccess">not in OA subset</error></OA>',
            )
        if (
            params.get("db") == "pmc"
            or "pubtator" in str(request.url)
            or path.endswith((".pdf", ".tar.gz"))
        ):
            protected_routes_called.append(str(request.url))
            raise AssertionError("non-OA full-text route must not be called")
        raise AssertionError(f"unexpected request: {request.url}")

    acquirer = LiteratureAcquirer(
        workspace,
        _settings(downloads),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        pdf_text_extractor=_pdf_text,
    )

    acquired = acquirer.acquire("12345678")

    assert acquired["status"] == "abstract_only"
    assert acquired["acquisition"]["markdown_status"] == "abstract_only"
    assert acquired["acquisition"]["pdf_path"] is None
    assert (
        acquired["acquisition"]["rights_status"]
        == "metadata_abstract_only_no_reuse_rights"
    )
    assert (
        "only PubMed metadata and abstract" in acquired["acquisition"]["terms_warning"]
    )
    markdown = Path(acquired["acquisition"]["markdown_path"]).read_text(
        encoding="utf-8"
    )
    assert "We report a verified effect" in markdown
    assert "## Results" not in markdown
    assert protected_routes_called == []


def test_managed_browser_import_is_path_confined_and_preserves_inbox(tmp_path: Path):
    workspace = tmp_path / "workspace"
    downloads = tmp_path / "downloads"
    workspace.mkdir()
    downloads.mkdir()
    browser_pdf = downloads / "article.pdf"
    browser_pdf.write_bytes(b"%PDF-1.7\n" + b"y" * 12_000)
    (downloads / "partial.pdf.crdownload").write_bytes(b"partial")
    acquirer = LiteratureAcquirer(
        workspace,
        _settings(downloads),
        client=_client(),
        pdf_text_extractor=_pdf_text,
    )

    listing = acquirer.list_browser_downloads()
    escaped = acquirer.import_browser_pdf("12345678", "../article.pdf")
    imported = acquirer.import_browser_pdf("12345678", "article.pdf")

    assert [item["filename"] for item in listing["pdfs"]] == ["article.pdf"]
    assert escaped["error"] == "BROWSER_PDF_IMPORT_FAILED"
    assert imported["status"] == "verified_manual_browser_pdf"
    assert imported["acquisition"]["rights_status"] == "private_user_provided"
    assert "does not grant reuse" in imported["acquisition"]["terms_warning"]
    assert browser_pdf.is_file()
    assert Path(imported["acquisition"]["pdf_path"]).is_file()
    assert "verified_manual_browser_pdf" in Path(
        imported["acquisition"]["markdown_path"]
    ).read_text(encoding="utf-8")


def test_wrong_article_pdf_fails_closed_without_final_artifact(tmp_path: Path):
    workspace = tmp_path / "workspace"
    downloads = tmp_path / "downloads"
    workspace.mkdir()
    downloads.mkdir()
    (downloads / "wrong.pdf").write_bytes(b"%PDF-1.7\n" + b"z" * 12_000)

    def unrelated_text(_path: Path, _pdftotext: Path) -> str:
        return "An unrelated document about geology and mineral surveys. " * 30

    acquirer = LiteratureAcquirer(
        workspace,
        _settings(downloads),
        client=_client(),
        pdf_text_extractor=unrelated_text,
    )

    result = acquirer.import_browser_pdf("12345678", "wrong.pdf")

    assert result["error"] == "BROWSER_PDF_IMPORT_FAILED"
    assert not list((workspace / "references" / "pdfs").glob("*.pdf"))


def test_pmc_archive_parser_rejects_path_traversal():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = b"<article/>"
        member = tarfile.TarInfo("../article.nxml")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))

    try:
        LiteratureAcquirer._nxml_from_archive(buffer.getvalue())
    except RuntimeError as exc:
        assert "no safe NXML" in str(exc)
    else:
        raise AssertionError("path traversal archive was accepted")


def _tar_gz(entries: list[tuple[tarfile.TarInfo, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for member, data in entries:
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


def test_pmc_archive_pdf_member_is_bounded_and_regular():
    pdf = b"%PDF-1.7\n" + b"x" * 12_000
    member = tarfile.TarInfo("article/main.pdf")
    archive = _tar_gz([(member, pdf)])

    extracted = list(LiteratureAcquirer._pdfs_from_archive(archive, len(pdf) + 1))

    assert extracted == [("article/main.pdf", pdf)]


@pytest.mark.parametrize("member_name", ["../escape.pdf", "/absolute.pdf"])
def test_pmc_archive_pdf_rejects_traversal(member_name: str):
    member = tarfile.TarInfo(member_name)
    archive = _tar_gz([(member, b"%PDF-1.7\n" + b"x" * 100)])

    with pytest.raises(RuntimeError, match="no safe PDF"):
        list(LiteratureAcquirer._pdfs_from_archive(archive, 10_000))


def test_pmc_archive_pdf_rejects_symlink():
    member = tarfile.TarInfo("article.pdf")
    member.type = tarfile.SYMTYPE
    member.linkname = "/etc/passwd"
    archive = _tar_gz([(member, b"")])

    with pytest.raises(RuntimeError, match="no safe PDF"):
        list(LiteratureAcquirer._pdfs_from_archive(archive, 10_000))


def test_pmc_archive_pdf_rejects_oversize_and_missing_pdf():
    oversized = tarfile.TarInfo("article.pdf")
    oversized_archive = _tar_gz([(oversized, b"%PDF-" + b"x" * 200)])
    text = tarfile.TarInfo("README.txt")
    no_pdf_archive = _tar_gz([(text, b"no pdf")])

    with pytest.raises(RuntimeError, match="exceeds"):
        list(LiteratureAcquirer._pdfs_from_archive(oversized_archive, 100))
    with pytest.raises(RuntimeError, match="no safe PDF"):
        list(LiteratureAcquirer._pdfs_from_archive(no_pdf_archive, 10_000))


def test_acquisition_falls_back_to_verified_pdf_inside_pmc_archive(tmp_path: Path):
    workspace = tmp_path / "workspace"
    downloads = tmp_path / "downloads"
    workspace.mkdir()
    downloads.mkdir()
    pdf = b"%PDF-1.7\n" + b"x" * 12_000
    nxml_member = tarfile.TarInfo("article/article.nxml")
    pdf_member = tarfile.TarInfo("article/article.pdf")
    archive_bytes = _tar_gz([(nxml_member, JATS_XML), (pdf_member, pdf)])
    archive_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal archive_requests
        path = request.url.path
        params = request.url.params
        if path.endswith("/efetch.fcgi") and params.get("db") == "pubmed":
            return httpx.Response(200, content=PUBMED_XML)
        if path.endswith("/efetch.fcgi") and params.get("db") == "pmc":
            return httpx.Response(404)
        if path.endswith("/oa.fcgi"):
            return httpx.Response(
                200,
                content=(
                    '<OA><records><record id="PMC9876543" license="CC BY" '
                    'retracted="no"><link format="pdf" '
                    'href="https://ftp.ncbi.nlm.nih.gov/pub/fake/missing.pdf" />'
                    '<link format="tgz" '
                    'href="https://ftp.ncbi.nlm.nih.gov/pub/fake/article.tar.gz" />'
                    "</record></records></OA>"
                ).encode(),
            )
        if path.endswith("/missing.pdf"):
            return httpx.Response(404)
        if path.endswith("/article.tar.gz"):
            archive_requests += 1
            return httpx.Response(200, content=archive_bytes)
        raise AssertionError(f"unexpected request: {request.url}")

    acquirer = LiteratureAcquirer(
        workspace,
        _settings(downloads),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        pdf_text_extractor=_pdf_text,
    )

    acquired = acquirer.acquire("12345678")

    assert acquired["status"] == "full_text_with_pdf"
    assert archive_requests == 1
    assert Path(acquired["acquisition"]["pdf_path"]).read_bytes() == pdf
    assert any(
        "archive member" in note
        for note in acquired["acquisition"]["acquisition_notes"]
    )


def test_pubmed_parser_does_not_invent_identifiers():
    record = _parse_pubmed_xml(PUBMED_XML)[0]
    assert record["pmid"] == "12345678"
    assert record["doi"] == "10.1000/verified.2024.1"
    assert record["canonical_url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    assert "isbn" not in json.dumps(record)


def test_pubmed_parser_ignores_nested_reference_identifiers():
    payload = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
    <PMID>28298962</PMID><Article><ArticleTitle>Canonical article</ArticleTitle>
    <Journal><Title>World Journal of Radiology</Title><JournalIssue><PubDate>
    <Year>2017</Year></PubDate></JournalIssue></Journal>
    <AuthorList><Author><LastName>Lee</LastName></Author></AuthorList></Article>
    </MedlineCitation><PubmedData><ArticleIdList>
    <ArticleId IdType="pubmed">28298962</ArticleId>
    <ArticleId IdType="pmc">PMC5334499</ArticleId>
    <ArticleId IdType="doi">10.4329/wjr.v9.i2.27</ArticleId>
    </ArticleIdList><ReferenceList><Reference><ArticleIdList>
    <ArticleId IdType="pmc">PMC5367446</ArticleId>
    <ArticleId IdType="doi">10.1000/wrong.reference</ArticleId>
    </ArticleIdList></Reference></ReferenceList></PubmedData>
    </PubmedArticle></PubmedArticleSet>"""

    record = _parse_pubmed_xml(payload)[0]

    assert record["pmid"] == "28298962"
    assert record["pmcid"] == "PMC5334499"
    assert record["doi"] == "10.4329/wjr.v9.i2.27"


def _literature_report(markdown: Path, pdf: Path | None = None) -> ScientificReport:
    return ScientificReport(
        title="Literature-backed report",
        executive_summary="A PubMed-indexed study reported a treatment effect.",
        introduction="Prior evidence was evaluated from an acquired article.",
        methods=["PubMed metadata and locally acquired article text were reviewed."],
        results="The acquired study reported a treatment effect.",
        discussion="Interpretation remains limited to the study design.",
        conclusions="The cited study supports the stated literature result.",
        claims=[
            ClaimRecord(
                claim_id="C1",
                text="The acquired study reported a treatment effect.",
                claim_type="literature_supported",
                evidence_refs=["S1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="S1",
                title="A verified treatment effect in a controlled study",
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                source_type="primary_study",
                retrieved_at="2026-07-14T12:00:00+00:00",
                supporting_passage="The abstract and article text report the effect.",
                doi="10.1000/verified.2024.1",
                pmid="12345678",
                pmcid="PMC9876543",
                citekey="lovelace-2024-pmid12345678",
                license="unknown",
                rights_status="pmc_oa_reuse_allowed",
                terms_warning="Automatically acquired under the recorded OA license.",
                retracted=False,
                local_pdf_path=str(pdf) if pdf else None,
                local_markdown_path=str(markdown),
                full_text_status="full_text_with_pdf"
                if pdf
                else "full_text_markdown_only",
            )
        ],
    )


def _acquisition_metadata(
    tmp_path: Path,
    markdown: Path,
    pdf: Path | None = None,
) -> Path:
    source = _literature_report(markdown, pdf).sources[0]
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    path = metadata_dir / f"{source.citekey}.json"
    path.write_text(
        json.dumps(
            {
                "article": {
                    "pmid": source.pmid,
                    "pmcid": source.pmcid,
                    "doi": source.doi,
                    "title": source.title,
                    "canonical_url": str(source.url),
                },
                "acquisition": {
                    "citekey": source.citekey,
                    "license": source.license,
                    "rights_status": source.rights_status,
                    "terms_warning": source.terms_warning,
                    "retracted": source.retracted,
                    "status": source.full_text_status,
                    "pdf_path": source.local_pdf_path,
                    "markdown_path": source.local_markdown_path,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_acquisition_metadata_resolves_by_pmid_when_model_citekey_is_missing_or_wrong(
    tmp_path: Path,
):
    markdown = tmp_path / "lovelace.md"
    markdown.write_text(
        "# Verified article\n\nThe controlled study reported a treatment effect.\n",
        encoding="utf-8",
    )
    report = _literature_report(markdown)
    metadata = _acquisition_metadata(tmp_path, markdown)
    retrieval = RetrievalEvidence(
        successful_calls=1,
        tools=["acquire_pubmed_article"],
        urls=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
        retrieval_dates=["2026-07-14"],
        artifacts=[str(markdown), str(metadata)],
    )

    for citekey in (None, "model-invented-citekey"):
        source = report.sources[0].model_copy(update={"citekey": citekey})
        acquired, resolved_metadata, acquired_markdown = load_acquired_article_record(
            source, retrieval
        )

        assert acquired["article"]["pmid"] == "12345678"
        assert resolved_metadata == metadata.resolve()
        assert "controlled study" in acquired_markdown

        validation = validate_report(
            report.model_copy(update={"sources": [source]}), retrieval=retrieval
        )
        codes = {finding.code for finding in validation.findings}
        assert "pubmed_acquisition_metadata_invalid" not in codes
        assert "pubmed_acquisition_metadata_mismatch" in codes


def test_local_literature_paths_are_linted_materialized_and_linked(tmp_path: Path):
    markdown = tmp_path / "lovelace.md"
    markdown.write_text(
        "# Verified article\n\nThe controlled study reported a treatment effect.\n",
        encoding="utf-8",
    )
    pdf = tmp_path / "lovelace.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 12_000)
    report = _literature_report(markdown, pdf)
    metadata = _acquisition_metadata(tmp_path, markdown, pdf)
    retrieval = RetrievalEvidence(
        successful_calls=1,
        tools=["acquire_pubmed_article"],
        urls=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
        retrieval_dates=["2026-07-14"],
        artifacts=[str(markdown), str(pdf), str(metadata)],
    )

    validation = validate_report(report, retrieval=retrieval)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = materialize_references(run_dir, report, retrieval)
    rendered = render_report_markdown(report, reference_manifest=manifest)

    assert validation.passed, validation.findings
    assert (
        run_dir / "references" / "pdfs" / "lovelace-2024-pmid12345678.pdf"
    ).is_file()
    assert "[PDF](references/pdfs/lovelace-2024-pmid12345678.pdf)" in rendered
    assert "[Markdown](references/markdown/lovelace-2024-pmid12345678.md)" in rendered
    assert "PMID 12345678" in rendered
    entry = manifest["references"][0]
    assert entry["rights_status"] == "pmc_oa_reuse_allowed"
    assert entry["terms_warning"] == report.sources[0].terms_warning


def test_precise_literature_number_must_exist_in_acquired_article(tmp_path: Path):
    markdown = tmp_path / "lovelace.md"
    markdown.write_text(
        "# Verified article\n\nThe controlled study reported a treatment effect.\n",
        encoding="utf-8",
    )
    report = _literature_report(markdown)
    report.claims[0].text = "The acquired study reported a 5% treatment effect."
    metadata = _acquisition_metadata(tmp_path, markdown)
    retrieval = RetrievalEvidence(
        successful_calls=1,
        tools=["acquire_pubmed_article"],
        urls=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
        retrieval_dates=["2026-07-14"],
        artifacts=[str(markdown), str(metadata)],
    )

    validation = validate_report(report, retrieval=retrieval)

    assert not validation.passed
    assert "literature_claim_number_not_grounded" in {
        finding.code for finding in validation.findings
    }


def test_required_pubmed_support_needs_search_acquisition_and_local_citation(
    tmp_path: Path,
):
    markdown = tmp_path / "article.md"
    markdown.write_text(
        "# Verified article\n\nThe controlled study reported a treatment effect.\n",
        encoding="utf-8",
    )
    report = _literature_report(markdown)
    metadata = _acquisition_metadata(tmp_path, markdown)
    complete = RetrievalEvidence(
        successful_calls=2,
        tools=["search_pubmed", "acquire_pubmed_article"],
        urls=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
        retrieval_dates=["2026-07-14"],
        artifacts=[str(markdown), str(metadata), str(metadata)],
    )

    assert validate_report(
        report,
        retrieval=complete,
        require_pubmed_literature=True,
    ).passed
    missing = validate_report(
        report.model_copy(update={"sources": []}),
        retrieval=RetrievalEvidence(),
        require_pubmed_literature=True,
    )
    assert {
        "pubmed_search_missing",
        "pubmed_article_not_acquired",
        "pubmed_source_not_cited",
    } <= {finding.code for finding in missing.findings}


def test_pubmed_source_identifier_mismatch_cannot_pass(tmp_path: Path):
    markdown = tmp_path / "article.md"
    markdown.write_text(
        "# Verified article\n\nThe controlled study reported a treatment effect.\n",
        encoding="utf-8",
    )
    report = _literature_report(markdown)
    metadata = _acquisition_metadata(tmp_path, markdown)
    report.sources[0] = report.sources[0].model_copy(
        update={"doi": "10.1000/fabricated"}
    )
    retrieval = RetrievalEvidence(
        successful_calls=2,
        tools=["search_pubmed", "acquire_pubmed_article"],
        urls=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
        retrieval_dates=["2026-07-14"],
        artifacts=[str(markdown), str(metadata)],
    )

    validation = validate_report(report, retrieval=retrieval)

    assert not validation.passed
    assert "pubmed_acquisition_metadata_mismatch" in {
        finding.code for finding in validation.findings
    }


def test_grossly_mismatched_literature_claim_cannot_pass(tmp_path: Path):
    markdown = tmp_path / "article.md"
    markdown.write_text(
        "# Verified article\n\nThe controlled study reported a treatment effect.\n",
        encoding="utf-8",
    )
    report = _literature_report(markdown)
    report.claims[0] = report.claims[0].model_copy(
        update={"text": "Lunar regolith definitively prevents melanoma metastasis."}
    )
    metadata = _acquisition_metadata(tmp_path, markdown)
    retrieval = RetrievalEvidence(
        successful_calls=2,
        tools=["search_pubmed", "acquire_pubmed_article"],
        urls=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
        retrieval_dates=["2026-07-14"],
        artifacts=[str(markdown), str(metadata)],
    )

    validation = validate_report(report, retrieval=retrieval)

    assert not validation.passed
    assert "literature_claim_not_lexically_grounded" in {
        finding.code for finding in validation.findings
    }


def test_controller_builds_bounded_acquired_passages_for_gemma(tmp_path: Path):
    markdown = tmp_path / "article.md"
    markdown.write_text(
        "# Verified article\n\n"
        "The controlled study reported a treatment effect with uncertainty.\n\n"
        + ("Unrelated appendix material. " * 1_000),
        encoding="utf-8",
    )
    report = _literature_report(markdown)
    metadata = _acquisition_metadata(tmp_path, markdown)
    retrieval = RetrievalEvidence(
        successful_calls=2,
        tools=["search_pubmed", "acquire_pubmed_article"],
        urls=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
        retrieval_dates=["2026-07-14"],
        artifacts=[str(markdown), str(metadata)],
    )

    packet = build_acquired_article_audit(report, retrieval)

    assert packet[0]["acquisition_metadata"]["article"]["pmid"] == "12345678"
    assert packet[0]["linked_claims"][0]["claim_id"] == "C1"
    assert "treatment effect" in packet[0]["controller_extracted_passages"][0]
    assert len(packet[0]["controller_extracted_passages"]) <= 4
    assert all(
        len(item) <= 1_500 for item in packet[0]["controller_extracted_passages"]
    )


@pytest.mark.asyncio
async def test_final_gemma_audit_receives_controller_extracted_article_bytes(
    tmp_path: Path,
    monkeypatch,
):
    from scientific_agent.orchestrator import _audit_report

    markdown = tmp_path / "article.md"
    markdown.write_text(
        "# Verified article\n\nThe controlled study reported a treatment effect.\n",
        encoding="utf-8",
    )
    report = _literature_report(markdown)
    metadata = _acquisition_metadata(tmp_path, markdown)
    retrieval = RetrievalEvidence(
        successful_calls=2,
        tools=["search_pubmed", "acquire_pubmed_article"],
        urls=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
        retrieval_dates=["2026-07-14"],
        artifacts=[str(markdown), str(metadata)],
    )
    task = TaskSpec(
        task_id="pubmed-audit",
        objective="Audit a biomedical claim",
        deliverables=["scientific report"],
        acceptance_tests=["claims grounded"],
    )
    plan = PlanProposal(
        plan_label="MASTER",
        objective=task.objective,
        steps=[
            PlanStep(
                step_id="S1",
                objective="Audit",
                outputs=["report"],
                methods=["PubMed"],
                validators=[
                    CheckSpec(
                        check_id="C1",
                        description="Ground claims",
                        check_type="source",
                    )
                ],
                stop_conditions=["Grounded"],
            )
        ],
        expected_artifacts=["scientific report"],
    )
    planning = PlanningResult(
        master_plan=MasterPlan(
            task=task,
            plan=plan,
            resolutions=[],
            method_lock_required=False,
        ),
        audit=VerificationReport(verdict="pass"),
        plan_lints=[],
        status="supported",
    )
    captured = {}

    async def fake_request(*_args, **kwargs):
        captured.update(kwargs["payload"])
        return VerificationReport(verdict="pass")

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )

    review = await _audit_report(
        Settings(),
        planning,
        report,
        DeterministicValidation(passed=True),
        retrieval,
        ComputationEvidence(),
    )

    assert review.verdict == "pass"
    evidence = captured["acquired_article_evidence"][0]
    assert evidence["acquisition_metadata"]["article"]["pmid"] == "12345678"
    assert "treatment effect" in evidence["controller_extracted_passages"][0]


def test_fabricated_local_literature_path_is_blocking(tmp_path: Path):
    markdown = tmp_path / "article.md"
    markdown.write_text("article", encoding="utf-8")
    report = _literature_report(markdown)
    retrieval = RetrievalEvidence(
        successful_calls=1,
        urls=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
        retrieval_dates=["2026-07-14"],
        artifacts=[],
    )

    validation = validate_report(report, retrieval=retrieval)

    assert not validation.passed
    assert "local_literature_artifact_not_retrieved" in {
        finding.code for finding in validation.findings
    }


def test_retracted_article_cannot_support_claim(tmp_path: Path):
    markdown = tmp_path / "article.md"
    markdown.write_text("article", encoding="utf-8")
    report = _literature_report(markdown)
    report.sources[0] = report.sources[0].model_copy(update={"retracted": True})
    retrieval = RetrievalEvidence(
        successful_calls=1,
        urls=["https://pubmed.ncbi.nlm.nih.gov/12345678/"],
        retrieval_dates=["2026-07-14"],
        artifacts=[str(markdown)],
    )

    validation = validate_report(report, retrieval=retrieval)

    assert "retracted_source_used_as_support" in {
        finding.code for finding in validation.findings
    }


def test_source_schema_rejects_claimed_pdf_without_path(tmp_path: Path):
    source = _literature_report(tmp_path / "article.md").sources[0]
    with pytest.raises(ValueError, match="requires a local PDF"):
        SourceRecord.model_validate(
            {**source.model_dump(), "full_text_status": "full_text_with_pdf"}
        )


def test_policy_records_only_path_confined_literature_artifacts(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    article = workspace / "article.md"
    article.write_text("verified", encoding="utf-8")
    ledger = EventLedger(tmp_path / "events.jsonl")
    policy = ToolPolicy(
        ledger=ledger,
        allowed_tools=default_allowed_tools(False),
        evidence_dir=tmp_path / "evidence",
        retrieval_artifact_roots=(workspace,),
    )
    tool = type("Tool", (), {"name": "acquire_pubmed_article"})()

    response = policy.after_tool(
        tool,
        {"pmid": "12345678"},
        None,
        {
            "status": "abstract_only",
            "article": {"canonical_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"},
            "artifacts": [str(article)],
        },
    )

    assert response is None
    evidence = policy.retrieval_evidence()
    assert str(article.resolve()) in evidence.artifacts
    assert evidence.urls == ["https://pubmed.ncbi.nlm.nih.gov/12345678/"]

    outside = tmp_path / "outside.md"
    outside.write_text("not workspace evidence", encoding="utf-8")
    denied = policy.after_tool(
        tool,
        {"pmid": "12345678"},
        None,
        {"status": "abstract_only", "artifacts": [str(outside)]},
    )
    assert denied["error"] == "INVALID_RETRIEVAL_ARTIFACT"
