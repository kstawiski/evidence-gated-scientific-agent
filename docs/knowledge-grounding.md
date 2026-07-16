# Knowledge grounding architecture

Evidence Bench uses an instance-local, operator-managed knowledge library. The
private and lab deployments never share a database or document volume: the
library lives below each deployment's `SCIENTIFIC_AGENT_DATA_DIR`.

## Scientific contract

Knowledge retrieval is evidence acquisition, not truth adjudication. Every
retrieved passage carries an immutable document-generation ID, extracted-text
SHA-256, index-version ID, chunk ID, exact character offsets, and a run-local
browser URL. The complete passage record and exact cited bytes are copied into the
run provenance. Qwen may use retrieved passages only after the method lock while
researching and drafting; Gemma audits the resulting claims. Deterministic checks
require each cited passage to exist in the run snapshot, recompute every content
and chunk hash, verify offsets against the immutable extracted text, and reject a
quoted passage that does not match those bytes. Paraphrases and precise numbers
receive lexical/numeric grounding checks. Model agreement never substitutes for
these checks.

On first use of a document in a run, the controller also copies the complete
extracted text and immutable original file into that run, verifies both against
the snapshotted hashes, and exposes them beside the exact passage in the WebUI.
Thus a cited knowledge paper remains previewable as full text and, when the
original was a PDF, as the PDF itself even after the live library generation is
retired.

Planning sees only a controller-owned, value-free manifest (titles, source types,
tags, counts, and hashes) before the method lock. Every metadata string is marked
as untrusted data, never an instruction. Planning never receives
knowledge passages or objective-matched results before the lock. A run snapshots
immutable document-generation IDs,
content hashes, metadata, and chunk-index versions so subsequent edits, disabling,
re-indexing, or logical deletion cannot change that run's evidence universe.
Uploaded visual inputs receive a separate Gemma-only structural intake before
planning. Outcome values, effect directions, significance, and other
result-bearing visual content are withheld until the protocol is locked; a fresh
post-lock Gemma review performs the scientific interpretation.

## Storage and retrieval

- SQLite metadata plus FTS5 lexical search using
  `unicode61 remove_diacritics 2`. Polish inflection and medical synonyms are
  addressed by multiple bounded query formulations, and recall@k is measured on
  a checked-in bilingual gold set; an absent hit is a retrieval limitation, never
  proof that evidence is absent.
- Immutable originals and extracted UTF-8 text under `knowledge/documents/<id>/`.
- Paragraph-aware, overlapping chunks with stable SHA-256 hashes and offsets.
- Exact lexical retrieval is the deterministic baseline. Qwen can issue multiple
  synonym/phrase searches through a typed read-only tool; no opaque embedding
  score is presented as scientific evidence.
- TXT, Markdown, CSV/TSV, JSON/YAML, PDF, DOCX, PPTX, and XLSX are accepted through
  bounded, fixed-code extractors. Unsupported or failed extraction is explicit.
- Document generations are immutable. Metadata edits and re-indexing create a new
  generation and retire the prior generation; existing runs keep using the exact
  generation they selected.
- Logical deletion removes a document from new runs and search while retaining
  immutable bytes needed by existing provenance. Re-imported changed content gets
  a new document ID.
- PubMed papers successfully acquired and validated during a run are auto-imported
  after completion. The controller copies the verified PDF/Markdown, deduplicates
  identical content/original hashes, groups changed content by PMID or DOI into a
  new immutable generation, and preserves access/rights metadata. Every successful
  acquisition is recorded once per source and run, including its workspace, run,
  identifiers, and hashes—even when its document bytes were already in the
  library. Search leads and failed or unverified browser downloads are never
  promoted.

## Web management

The Knowledge Library dialog supports:

1. upload with title, description, tags, source type, and optional canonical URL;
2. list/filter with enabled, indexing, size, chunk, and checksum status;
3. source download, extracted-text preview, chunk inspection, and verified run
   import history;
4. metadata edit and enable/disable;
5. deterministic re-index of one or all documents;
6. retrieval test showing exact ranked passages and offsets;
7. logical deletion with a provenance-retention warning;
8. per-run document selection, defaulting to all enabled documents.

The API exposes the same operations for automation. Concurrent changes are safe
because additions cannot enter an existing snapshot and edits/re-indexing create
new immutable generations. Disable/delete changes affect only future selections.
Every search is filtered to the snapshot's exact document-generation IDs and
index versions, then re-verifies text and chunk hashes before returning a hit.
FTS and metadata changes commit in one SQLite transaction; a crash cannot expose a
partially indexed generation. Optimistic revision tokens reject clobbering edits.

## Security and privacy

- Filenames are basenames only; symlinks and traversal are rejected.
- Upload, extracted-text, document, chunk, and result sizes are bounded.
- Archive-based formats are read without extracting paths to disk.
- Archive member count, total uncompressed bytes, individual member bytes, and
  compression ratio are bounded before content is decoded.
- PDF conversion uses a fixed `pdftotext` invocation with no shell.
- Search is parameterized and FTS query terms are controller-generated.
- Retrieved text is carried only in an `untrusted_source_text` JSON field, never
  concatenated with controller instructions. Source text is escaped as data and
  cannot add tools, change policy, or provide a passing audit by instruction.
- Source files are never sent to Brave, Context7, or the managed browser.
- Passwordless internal deployment does not change path confinement or A2A token
  isolation.
- Each library is stamped with a deployment identity. Startup refuses a data
  directory stamped for another Compose instance, preventing accidental private/
  lab volume sharing.

## Acceptance criteria

- CRUD, preview/download, enable/disable, re-index, search, and run selection pass
  API and browser tests.
- A seeded passage is retrieved with the right document/chunk/hash/offset and is
  preserved in the run bundle.
- Disabled/deleted/unselected documents cannot be retrieved by a new run.
- A run's selected hashes cannot change after submission.
- Re-index/edit creates a new generation; an old run still resolves identical
  passage bytes and hashes from its provenance after the live generation changes.
- A fabricated chunk ID, changed quote, wrong offset/hash, or post-submission
  document is deterministically rejected.
- A verified acquired PubMed paper produces one current content generation and
  one acquisition record per source/run; a search result or failed/manual file
  without controller verification is not imported.
- Prompt injection inside a document is returned as quoted source content and is
  explicitly marked untrusted; it cannot add tools, alter controller policy, or
  flip deterministic claim support.
- Both port 80 and port 8070 retain independent libraries across container restart.
- Concurrent workspaces and a killed re-index cannot alter a submitted snapshot or
  leave a partially searchable active generation.
- Retrieval evaluation reports recall@k for English/Polish spelling, diacritics,
  inflection, abbreviations, and medical synonyms.

Run citations resolve to hash-checked files below that run's provenance directory,
not the mutable library API. A completed bundle therefore keeps the cited passage
and metadata offline after the source generation is retired or logically deleted.
Character offsets are offsets in the immutable extracted text, not the original
PDF byte stream; flattened PDF tables are unsuitable for cell-level claims unless
they are separately extracted into a structured table.
