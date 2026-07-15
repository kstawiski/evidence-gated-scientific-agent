# PubMed and full-text acquisition

Evidence Bench exposes five typed literature tools to Qwen:

- `search_pubmed(query, max_results)` searches PubMed through NCBI E-utilities
  and stores the returned metadata under `references/searches/`.
- `acquire_pubmed_article(pmid)` resolves the exact PubMed record, obtains
  legitimate open-access article text where available, and records explicit
  access states.
- `search_acquired_article(citekey, query, max_matches)` returns bounded matching
  passages from one locally acquired Markdown article without exposing an
  arbitrary file-read primitive.
- `list_browser_downloads()` lists only direct, regular PDF basenames in the
  service-owned browser inbox.
- `import_browser_downloaded_pdf(pmid, filename)` imports one regular PDF from
  the service-owned browser inbox at `/browser-downloads`. It accepts a basename,
  never an arbitrary path, and leaves the inbox copy untouched.

The tools do not use a user's personal Chrome instance, institutional proxy, or
private subscription. Network requests are restricted to fixed NCBI/PMC hosts.
The HTTP client ignores proxy environment variables and follows redirects only
through a bounded controller loop that revalidates HTTPS, URL credentials, and
the exact host allow-list at every hop. Redirects to private IPs or other domains
fail closed.
Configure a real maintainer contact as `SCIENTIFIC_AGENT_NCBI_EMAIL`; Evidence
Bench supplies that email and `SCIENTIFIC_AGENT_NCBI_TOOL` on E-utilities calls
and remains below three requests per second unless an optional NCBI API key is
configured.

For biomedical, clinical, health, life-science, and medical analyses this is a
quality gate, not an optional convenience: the run must record a PubMed search,
acquire at least one relevant article, and cite its local Markdown evidence copy.
The article supplies context and methodological support; it never substitutes
for analyzing the user's data. Pure software-engineering and non-biomedical
tasks are not forced through PubMed.

## Stored collection

Each isolated workspace owns its collection:

```text
references/
├── bibliography.md
├── searches/pubmed-<query-hash>.json
├── metadata/<citekey>.json
├── markdown/<citekey>.md
└── pdfs/<citekey>.pdf                 # only when verified and available
```

The canonical PMID, PMCID, and DOI remain in metadata and the report's
`SourceRecord`; local files supplement rather than replace those identifiers.
Only the record-level PubMed `PubmedData/ArticleIdList` is authoritative; IDs in
nested reference lists cannot override the requested article's identifiers.
The final provenance bundle copies cited local files into the same relative
`references/markdown/` and `references/pdfs/` paths and writes
`reference_manifest.json`, including hashes, byte sizes, license, rights status,
and the applicable terms warning. Report citations link
to the browser-previewable Markdown/PDF copies plus the canonical PubMed record.
In the WebUI, selecting a locally acquired source opens Markdown in the large
text preview; a verified PDF opens inline through the hash-checked
`/api/runs/{run_id}/references/{source_id}/pdf` route.

## Acquisition and verification

The automatic route is deliberately conservative:

1. verify the PubMed metadata returned for the requested numeric PMID;
2. resolve PMCID/DOI only from that record;
3. require an exact-PMCID record from the PMC Open Access subset with both an
   explicit reusable license (CC BY-family, CC0, or public domain) and an
   allow-listed OA PDF/TGZ route;
4. only after that rights gate succeeds, request PMC JATS XML through E-utilities;
5. fall back to a safe NXML member from the PMC OA archive and then PubTator3
   full text;
6. retain only a locally searchable PubMed metadata/abstract record, with an
   explicit unavailability reason, when the rights gate fails;
7. try a PMC OA PDF link when supplied, while treating PDF absence as an explicit
   state rather than an error or invented file.

A PMCID alone is not a copyright or redistribution permission. When the OA gate
fails, Evidence Bench never calls PMC XML, PubTator full text, or an article
download route for that record. Manually imported browser PDFs are instead marked
`private_user_provided`: they are user-provided private artifacts, not evidence of
open-access status, and their warning requires users to verify publisher and
institutional terms before any redistribution.

If PMC's individual PDF link is absent or stale, the same already-downloaded OA
archive is inspected for a regular PDF member. Traversal paths, absolute paths,
symlinks, hardlinks, oversized members, and excessive member counts are rejected;
the extracted bytes still undergo the normal PDF signature, size, text, and
article-identity verification before they can be stored.

PMC has announced that individual OA PDF FTP downloads will end in August 2026,
so the workflow does not depend on that route. NXML/Markdown is the durable
machine-readable evidence copy; PDF is optional.

Every PDF is size-bounded, must contain a PDF signature, must yield meaningful
text through `pdftotext`, and must match the expected DOI/PMID/PMCID or article
title. A login page, bot-check page, wrong article, malformed file, or unverifiable
scan fails closed. TGZ extraction rejects absolute paths, traversal members,
links, and oversized NXML files.

The application process never invokes Poppler. It sends only an absolute PDF path
and a fresh request UUID to the token-authenticated sandbox worker. The worker
accepts exactly one regular, non-symlink file at
`/data/workspaces/<uuid>/files/references/pdfs/<name>.pdf`, then launches
`pdftotext` in a fresh `bwrap --unshare-all` namespace. That namespace has no
network, sees only the one PDF read-only and one private temporary output directory,
and receives only the Poppler binary, required library/font data, `/dev/null`, and
bounded CPU/memory/process/file resources. Local direct parsing fails closed.

Acquisition states are one of:

- `full_text_with_pdf`
- `full_text_markdown_only`
- `abstract_only`
- `verified_manual_browser_pdf`
- `unavailable`

`abstract_only` must never be described as full text. Missing or inaccessible
full text limits the claims the agent may support.

Before a report can pass, deterministic validation compares every PubMed
`SourceRecord` identifier, citekey, acquisition status, license/retraction flag,
rights status, terms warning, and local path against the stored acquisition
metadata. It also rejects grossly
unrelated literature claims with no informative lexical overlap. The final Gemma
audit receives bounded passages read by the controller from the acquired Markdown
plus the matching acquisition metadata and independently evaluates semantic
entailment; a model-authored supporting paraphrase is not treated as source text.

## NCBI and copyright notice

PubMed/PMC metadata and content are provided by NCBI. NCBI does not endorse
Evidence Bench or downstream analyses. Availability in PubMed or PMC does not
waive copyright: license and retraction metadata are retained with every article,
and users must comply with the terms attached to each work. A retracted article
must be identified prominently and cannot be treated as ordinary supporting
evidence.
