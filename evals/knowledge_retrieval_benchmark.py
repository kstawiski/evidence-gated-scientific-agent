#!/usr/bin/env python3
"""Compare exact lexical retrieval with live Qwen descriptor enrichment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw

from scientific_agent import __version__
from scientific_agent.config import Settings
from scientific_agent.knowledge import KnowledgeLibrary
from scientific_agent.knowledge_indexing import KnowledgeSemanticIndexer
from scientific_agent.knowledge_visuals import register_document_visuals
from scientific_agent.provenance import utc_now


def _build_corpus(records: list[dict]) -> str:
    sections = []
    for record in records:
        marker = f"[DOC:{record['id']}]"
        base = f"{marker}\n{record['title']}\n{record['text']}\n"
        boundary = (
            f"Scope boundary for {record['id']}: this record must remain distinct "
            "from unrelated scientific records and does not support unstated claims. "
        )
        sections.append((base + boundary * 20)[:2_050])
    return "\n\n".join(sections)


def _rank(result: dict, relevant: str, limit: int) -> int | None:
    marker = f"[DOC:{relevant}]"
    for index, passage in enumerate(result["passages"][:limit], start=1):
        if marker in passage["untrusted_source_text"]:
            return index
    return None


def _evaluate(library, snapshot, queries, limit: int = 10) -> list[dict]:
    rows = []
    for item in queries:
        result = library.search(item["query"], snapshot, limit)
        rank = _rank(result, item["relevant"], limit)
        rows.append(
            {
                **item,
                "rank": rank,
                "recall_at_10": float(rank is not None),
                "ndcg_at_10": 0.0 if rank is None else 1.0 / math.log2(rank + 1),
                "methods": sorted(
                    {passage["retrieval_method"] for passage in result["passages"]}
                ),
            }
        )
    return rows


def _visual_presentation(path: Path, visuals: list[dict]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            "<slide><title>Scientific visual retrieval benchmark</title></slide>",
        )
        for index, item in enumerate(visuals, start=1):
            image = Image.new("RGB", (1_200, 800), "white")
            draw = ImageDraw.Draw(image)
            draw.text((90, 45), item["label"], fill="black", font_size=34)
            color = item["color"]
            kind = item["kind"]
            if kind == "survival":
                draw.line((130, 650, 1_080, 650), fill="black", width=4)
                draw.line((130, 130, 130, 650), fill="black", width=4)
                draw.text((470, 690), "Time (months)", fill="black", font_size=28)
                draw.text((145, 110), "Event-free proportion", fill="black", font_size=25)
                draw.line(
                    [(150, 180), (350, 180), (350, 260), (590, 260), (590, 390), (810, 390), (810, 520), (1050, 520)],
                    fill=color,
                    width=8,
                )
                draw.line(
                    [(150, 180), (280, 180), (280, 330), (500, 330), (500, 470), (740, 470), (740, 590), (1050, 590)],
                    fill="gray",
                    width=8,
                )
                for x, y in ((450, 260), (690, 390), (910, 520)):
                    draw.line((x, y - 12, x, y + 12), fill=color, width=4)
                draw.text((820, 120), "Group 1", fill=color, font_size=25)
                draw.text((820, 155), "Group 2", fill="gray", font_size=25)
            elif kind == "forest":
                draw.line((650, 130, 650, 670), fill="gray", width=5)
                draw.text((525, 690), "Effect estimate (95% CI)", fill="black", font_size=27)
                for row, (center, spread) in enumerate(((480, 130), (590, 170), (720, 110), (790, 150)), start=1):
                    y = 150 + row * 105
                    draw.text((110, y - 16), f"Study {row}", fill="black", font_size=25)
                    draw.line((center - spread, y, center + spread, y), fill=color, width=7)
                    draw.rectangle((center - 11, y - 11, center + 11, y + 11), fill=color)
                draw.text((585, 100), "Null", fill="gray", font_size=24)
            elif kind == "volcano":
                draw.line((130, 650, 1_080, 650), fill="black", width=4)
                draw.line((600, 120, 600, 650), fill="gray", width=3)
                draw.text((440, 690), "log2 fold change", fill="black", font_size=28)
                draw.text((145, 110), "-log10 adjusted p", fill="black", font_size=25)
                points = [(220, 270), (280, 390), (360, 480), (470, 560), (540, 600), (650, 590), (730, 510), (820, 410), (940, 250), (1010, 330)]
                for x, y in points:
                    draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color if x < 380 or x > 820 else "gray")
            elif kind == "flow":
                boxes = [
                    (410, 110, 790, 200, "Assessed for eligibility"),
                    (410, 265, 790, 355, "Randomized"),
                    (120, 430, 500, 520, "Allocated to group 1"),
                    (700, 430, 1_080, 520, "Allocated to group 2"),
                    (120, 600, 500, 690, "Included in analysis"),
                    (700, 600, 1_080, 690, "Included in analysis"),
                ]
                for left, top, right, bottom, label in boxes:
                    draw.rectangle((left, top, right, bottom), outline=color, width=5)
                    draw.text((left + 20, top + 28), label, fill="black", font_size=24)
                for line in ((600, 200, 600, 265), (600, 355, 310, 430), (600, 355, 890, 430), (310, 520, 310, 600), (890, 520, 890, 600)):
                    draw.line(line, fill=color, width=5)
            elif kind == "agreement":
                draw.line((130, 650, 1_080, 650), fill="black", width=4)
                draw.text((500, 690), "Average", fill="black", font_size=28)
                draw.text((145, 110), "Difference", fill="black", font_size=25)
                for y, label, width in ((250, "Upper limit", 4), (400, "Mean bias", 6), (550, "Lower limit", 4)):
                    draw.line((150, y, 1_050, y), fill=color if y == 400 else "gray", width=width)
                    draw.text((900, y - 35), label, fill="black", font_size=22)
                for x, y in ((210, 430), (300, 360), (390, 470), (480, 390), (570, 330), (660, 440), (760, 380), (850, 460)):
                    draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)
            elif kind == "calibration":
                draw.line((130, 650, 1_080, 650), fill="black", width=4)
                draw.line((130, 130, 130, 650), fill="black", width=4)
                draw.text((440, 690), "Predicted probability", fill="black", font_size=28)
                draw.text((145, 110), "Observed proportion", fill="black", font_size=25)
                draw.line((150, 630, 1_030, 160), fill="gray", width=4)
                draw.line([(150, 620), (320, 560), (500, 470), (690, 350), (860, 280), (1_030, 190)], fill=color, width=8)
                draw.text((835, 120), "Ideal", fill="gray", font_size=24)
                draw.text((835, 155), "Model", fill=color, font_size=24)
            image_path = path.parent / f"visual-{index}.png"
            image.save(image_path)
            archive.write(image_path, f"ppt/media/image{index}.png")
            image_path.unlink()


async def _evaluate_visuals(
    library: KnowledgeLibrary, indexer: KnowledgeSemanticIndexer, case: dict, root: Path
) -> dict:
    presentation = root / "visual-benchmark.pptx"
    _visual_presentation(presentation, case["visuals"])
    with presentation.open("rb") as handle:
        document = library.ingest(
            presentation.name,
            handle,
            presentation.stat().st_size,
            title="Scientific visual retrieval benchmark",
            source_type="dataset",
            semantic_pending=True,
        )
    assets = register_document_visuals(library, document["id"])
    print("benchmark: indexing six exact visual assets with Gemma", file=sys.stderr, flush=True)
    started = time.monotonic()
    await indexer.index_document(library, document["id"])
    snapshot = library.snapshot([document["id"]])
    rows = []
    for index, item in enumerate(case["visuals"], start=1):
        result = library.search_visuals(item["query"], snapshot, 5)
        expected = f"ppt/media/image{index}.png"
        rank = next(
            (
                position
                for position, visual in enumerate(result["visuals"], start=1)
                if visual["source_label"] == expected
            ),
            None,
        )
        rows.append({**item, "expected": expected, "rank": rank})
    recall = mean(float(row["rank"] is not None) for row in rows)
    top1 = mean(float(row["rank"] == 1) for row in rows)
    ndcg = mean(
        0.0 if row["rank"] is None else 1.0 / math.log2(row["rank"] + 1)
        for row in rows
    )
    no_answer_rows = []
    for query in case.get("visual_no_answer_queries", []):
        result = library.search_visuals(query, snapshot, 5)
        no_answer_rows.append(
            {"query": query, "hits": len(result["visuals"])}
        )
    no_answer_false_positive_rate = mean(
        float(row["hits"] > 0) for row in no_answer_rows
    )
    return {
        "document_id": document["id"],
        "assets": len(assets),
        "recall_at_5": recall,
        "top1_accuracy": top1,
        "ndcg_at_5": ndcg,
        "no_answer_false_positive_rate": no_answer_false_positive_rate,
        "indexing_seconds": time.monotonic() - started,
        "passed": (
            len(assets) == len(case["visuals"])
            and recall >= 0.85
            and top1 >= 2 / 3
            and ndcg >= 0.75
            and no_answer_false_positive_rate == 0
        ),
        "queries": rows,
        "no_answer_queries": no_answer_rows,
    }


def _bootstrap_difference(
    lexical: list[dict], hybrid: list[dict], strata: set[str], samples: int = 10_000
) -> dict:
    deltas = [
        candidate["recall_at_10"] - baseline["recall_at_10"]
        for baseline, candidate in zip(lexical, hybrid, strict=True)
        if baseline["stratum"] in strata
    ]
    generator = random.Random(20260716)
    bootstrap = sorted(
        mean(generator.choices(deltas, k=len(deltas))) for _ in range(samples)
    )
    return {
        "absolute_difference": mean(deltas),
        "bootstrap_95_ci": [bootstrap[249], bootstrap[9749]],
        "queries": len(deltas),
    }


async def run_benchmark(fixture: Path) -> dict:
    case = json.loads(fixture.read_text(encoding="utf-8"))
    corpus = _build_corpus(case["records"])
    with tempfile.TemporaryDirectory(prefix="evidence-knowledge-benchmark-") as raw:
        library = KnowledgeLibrary(
            Path(raw) / "knowledge", "retrieval-benchmark", "https://benchmark.invalid"
        )
        document = library.ingest(
            "benchmark.md",
            BytesIO(corpus.encode()),
            len(corpus.encode()),
            title="Bilingual scientific retrieval benchmark",
            source_type="dataset",
        )
        lexical_snapshot = library.snapshot([document["id"]])
        lexical = _evaluate(library, lexical_snapshot, case["queries"])
        settings = Settings()
        indexer = KnowledgeSemanticIndexer(settings)
        candidate = library.reindex(
            document["id"], document["etag"], semantic_pending=True
        )
        selected_chunks = library.semantic_source_chunks(candidate["id"])
        print(
            f"benchmark: indexing {len(selected_chunks)} exact text chunks with Qwen",
            file=sys.stderr,
            flush=True,
        )
        started = time.monotonic()
        await indexer.index_document(
            library,
            candidate["id"],
            expected_etag=candidate["etag"],
            expected_previous_document_id=document["id"],
        )
        text_indexing_seconds = time.monotonic() - started
        hybrid_snapshot = library.snapshot([candidate["id"]])
        hybrid = _evaluate(library, hybrid_snapshot, case["queries"])
        immutable_baseline_preserved = bool(
            library.search(case["queries"][0]["query"], lexical_snapshot, 10)[
                "passages"
            ]
        )
        synonym_gain = _bootstrap_difference(
            lexical, hybrid, {"synonym", "polish"}
        )
        exact_delta = _bootstrap_difference(lexical, hybrid, {"exact"})
        summary = {
            "lexical": {
                "recall_at_10": mean(row["recall_at_10"] for row in lexical),
                "ndcg_at_10": mean(row["ndcg_at_10"] for row in lexical),
            },
            "hybrid": {
                "recall_at_10": mean(row["recall_at_10"] for row in hybrid),
                "ndcg_at_10": mean(row["ndcg_at_10"] for row in hybrid),
            },
            "synonym_and_polish_recall_gain": synonym_gain,
            "exact_recall_delta": exact_delta,
        }
        thresholds = {
            "hybrid_recall_at_10_at_least_0_90": summary["hybrid"][
                "recall_at_10"
            ]
            >= 0.90,
            "hybrid_ndcg_at_10_at_least_0_75": summary["hybrid"]["ndcg_at_10"]
            >= 0.75,
            "vocabulary_gap_gain_at_least_0_10": synonym_gain[
                "absolute_difference"
            ]
            >= 0.10,
            "vocabulary_gap_gain_ci_excludes_zero": synonym_gain[
                "bootstrap_95_ci"
            ][0]
            > 0,
            "exact_recall_regression_at_most_0_01": exact_delta[
                "absolute_difference"
            ]
            >= -0.01,
            "published_baseline_snapshot_remains_searchable": immutable_baseline_preserved,
        }
        no_answer_rows = []
        descriptor_evidence_violations = 0
        for query in case.get("no_answer_queries", []):
            lexical_result = library.search(query, lexical_snapshot, 10)
            hybrid_result = library.search(query, hybrid_snapshot, 10)
            for passage in hybrid_result["passages"]:
                if "search_text" in passage or "descriptor" in passage:
                    descriptor_evidence_violations += 1
            no_answer_rows.append(
                {
                    "query": query,
                    "lexical_hits": len(lexical_result["passages"]),
                    "hybrid_hits": len(hybrid_result["passages"]),
                }
            )
        no_answer_false_positive_rate = mean(
            float(item["hybrid_hits"] > 0) for item in no_answer_rows
        )
        thresholds["no_answer_false_positive_rate_is_zero"] = (
            no_answer_false_positive_rate == 0
        )
        thresholds["descriptor_prose_never_returned_as_evidence"] = (
            descriptor_evidence_violations == 0
        )
        visual = await _evaluate_visuals(library, indexer, case, Path(raw))
        thresholds["visual_asset_extraction_complete"] = (
            visual["assets"] == len(case["visuals"])
        )
        thresholds["visual_recall_at_5_at_least_0_85"] = (
            visual["recall_at_5"] >= 0.85
        )
        thresholds["visual_top1_accuracy_at_least_0_67"] = (
            visual["top1_accuracy"] >= 2 / 3
        )
        thresholds["visual_ndcg_at_5_at_least_0_75"] = visual["ndcg_at_5"] >= 0.75
        thresholds["visual_no_answer_false_positive_rate_is_zero"] = (
            visual["no_answer_false_positive_rate"] == 0
        )
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            git_dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            git_commit, git_dirty = "unavailable", None
        return {
            "benchmark_version": 2,
            "generated_at": utc_now(),
            "application_version": __version__,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "models": {
                "text": settings.qwen.model,
                "visual": settings.gemma.model,
                "qwen_native_json_schema": settings.qwen.native_json_schema,
                "qwen_max_tokens": settings.qwen.max_tokens,
                "gemma_max_tokens": settings.gemma.max_tokens,
            },
            "fixture": str(fixture),
            "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "document_id": candidate["id"],
            "baseline_document_id": document["id"],
            "semantic_metadata": library.get_document(candidate["id"])[
                "semantic_metadata"
            ],
            "text_indexing_seconds": text_indexing_seconds,
            "summary": summary,
            "thresholds": thresholds,
            "passed": all(thresholds.values()),
            "visual": visual,
            "no_answer": {
                "false_positive_rate": no_answer_false_positive_rate,
                "descriptor_evidence_violations": descriptor_evidence_violations,
                "queries": no_answer_rows,
            },
            "queries": [
                {"lexical": base, "hybrid": candidate}
                for base, candidate in zip(lexical, hybrid, strict=True)
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).parent / "cases" / "knowledge_retrieval.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run_benchmark(args.fixture))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
