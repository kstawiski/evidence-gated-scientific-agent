import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scientific_agent.linting import validate_report
from scientific_agent.provenance import sha256_file
from scientific_agent.reporting import (
    _figure_layout_review_questions,
    _parse_tesseract_tsv,
    describe_available_displays,
    figure_annotation_overlap_candidates,
    inspect_figure,
    materialize_displays,
    prepare_display_audit,
    render_report_markdown,
)
from scientific_agent.results_workbook import build_results_workbook
from scientific_agent.schemas import (
    ArtifactRef,
    ClaimRecord,
    ComputationEvidence,
    ComputationRecord,
    ReportDisplay,
    ScientificReport,
)


def _fixture(tmp_path: Path):
    figures = tmp_path / "computations" / "exec-001" / "output" / "figures"
    tables = tmp_path / "computations" / "exec-001" / "output" / "tables"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)
    figure = figures / "effect.png"
    Image.new("RGB", (640, 400), color=(232, 244, 242)).save(figure)
    table = tables / "effects.csv"
    table.write_text("group,estimate\nA|B,1.25\ncontrol,0.00\n", encoding="utf-8")
    artifacts = [
        ArtifactRef(
            path=str(path.resolve()),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (figure, table)
    ]
    computation = ComputationEvidence(successful_calls=1, artifacts=artifacts)
    report = ScientificReport(
        title="Effect analysis",
        executive_summary="The exploratory estimate was 1.25 units.",
        introduction="The analysis evaluated a prespecified group contrast.",
        methods=["A reproducible grouped analysis was run in the sandbox."],
        results="Figure 1 shows the estimate and Table 1 reports its exact value.",
        discussion="The estimate is exploratory and requires external validation.",
        conclusions="The observed contrast warrants confirmation in new data.",
        displays=[
            ReportDisplay(
                display_id="effect-plot",
                kind="figure",
                title="Estimated group contrast",
                caption="Points show group estimates; bars denote 95% confidence intervals.",
                artifact_path=str(figure.resolve()),
                alt_text=(
                    "Point plot of group estimates with group on the x-axis and "
                    "estimated value on the y-axis; group A is higher than control."
                ),
            ),
            ReportDisplay(
                display_id="effect-table",
                kind="table",
                title="Exact group estimates",
                caption="Rows report the exploratory estimate for each analysis group.",
                artifact_path=str(table.resolve()),
            ),
        ],
        claims=[],
        sources=[],
        limitations=["The synthetic fixture has no external validity."],
    )
    return report, computation


def test_registered_displays_validate_and_render_portably(tmp_path: Path):
    report, computation = _fixture(tmp_path)

    validation = validate_report(report, computation=computation)
    assert validation.passed, validation.findings

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = materialize_displays(run_dir, report, computation)
    markdown = render_report_markdown(report, manifest)

    assert [item["number"] for item in manifest["displays"]] == [1, 1]
    assert manifest["displays"][0]["width"] == 640
    assert manifest["displays"][1]["rows"][0] == ["A|B", "1.25"]
    headings = [
        "## Abstract",
        "## Introduction",
        "## Methods",
        "## Results",
        "## Discussion",
        "## Conclusions",
        "## Evidence ledger",
        "## Sources",
    ]
    assert [markdown.index(item) for item in headings] == sorted(
        markdown.index(item) for item in headings
    )
    assert "![Point plot" in markdown
    assert "A\\|B" in markdown
    assert str(tmp_path) not in markdown


def test_provisional_displays_remain_available_without_mutating_run(tmp_path: Path):
    report, computation = _fixture(tmp_path)

    manifest = describe_available_displays(
        report,
        computation,
        validated=False,
        quality_status="requires_human_decision",
    )

    assert manifest["validated"] is False
    assert manifest["virtual"] is True
    assert manifest["omissions"] == []
    assert [item["kind"] for item in manifest["displays"]] == ["figure", "table"]
    assert all(item["path"] is None for item in manifest["displays"])
    assert not (tmp_path / "run").exists()


def test_results_workbook_contains_claim_ledger_and_full_tables(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    table_display = report.displays[1]
    table_path = Path(table_display.artifact_path)
    content = build_results_workbook(
        report,
        [
            (
                {"display_id": table_display.display_id, "title": table_display.title},
                table_path,
            )
        ],
        run_id="run-1234",
        quality_status="requires_human_decision",
    )
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(content)

    assert content.startswith(b"PK")
    with zipfile.ZipFile(workbook) as archive:
        archive.testzip()
        workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
        sheets = "".join(
            archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet")
        )
    assert 'name="README"' in workbook_xml
    assert 'name="Claims"' in workbook_xml
    assert "Exact group estimates" in workbook_xml
    assert "requires_human_decision" in sheets
    assert "A|B" in sheets


def test_unvalidated_markdown_never_presents_model_claim_labels_as_gate_status(
    tmp_path,
):
    report, _ = _fixture(tmp_path)
    report = report.model_copy(
        update={
            "claims": [
                ClaimRecord(
                    claim_id="provisional",
                    text="The provisional estimate was 1.25 units.",
                    claim_type="computed",
                    evidence_refs=[],
                    status="supported",
                )
            ]
        }
    )

    markdown = render_report_markdown(
        report,
        quality_status="requires_human_decision",
    )

    assert "NOT VALIDATED" in markdown
    assert "requires_human_decision" in markdown
    assert "model-labeled supported; run not validated" in markdown


def test_tesseract_tsv_parser_returns_bounded_text_and_geometry():
    raw = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t12\t24\t80\t20\t96.5\tHedges\n"
        "5\t1\t1\t1\t1\t2\t98\t24\t12\t20\t95.0\tg\n"
    ).encode()

    result = _parse_tesseract_tsv(raw)

    assert result["available"] is True
    assert result["text"] == "Hedges g"
    assert result["words"][0] == {
        "text": "Hedges",
        "confidence": 96.5,
        "left": 12,
        "top": 24,
        "width": 80,
        "height": 20,
    }


def test_layout_review_questions_prioritize_top_overlap_and_legend_data(
    tmp_path: Path,
):
    image_path = tmp_path / "overlap.png"
    image = Image.new("RGB", (1000, 600), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((800, 130, 930, 250), fill=(20, 90, 210))
    image.save(image_path)
    ocr = {
        "available": True,
        "words": [
            {
                "text": "Primary",
                "confidence": 96,
                "left": 330,
                "top": 20,
                "width": 220,
                "height": 60,
            },
            {
                "text": "Analysis",
                "confidence": 95,
                "left": 390,
                "top": 35,
                "width": 250,
                "height": 65,
            },
            {
                "text": "Mean",
                "confidence": 93,
                "left": 430,
                "top": 50,
                "width": 180,
                "height": 55,
            },
            {
                "text": "Treatment",
                "confidence": 98,
                "left": 700,
                "top": 120,
                "width": 130,
                "height": 28,
            },
            {
                "text": "Control",
                "confidence": 98,
                "left": 700,
                "top": 160,
                "width": 115,
                "height": 28,
            },
            {
                "text": "Observations",
                "confidence": 98,
                "left": 700,
                "top": 200,
                "width": 165,
                "height": 28,
            },
            {
                "text": "Group",
                "confidence": 98,
                "left": 700,
                "top": 240,
                "width": 95,
                "height": 28,
            },
        ],
    }

    questions = _figure_layout_review_questions(
        image_path,
        ocr,
        width=1000,
        height=600,
    )

    assert questions["pixel_interpretation_authority"] == "Gemma"
    top = questions["top_text_clearance"]
    assert top["candidate_overlap_count_in_top_22_percent"] >= 2
    assert top["priority"] == "high"
    assert top["examples"]
    assert all(len(item["union_box_fraction"]) == 4 for item in top["examples"])
    legend = questions["legend_data_clearance"]["candidate"]
    assert legend["priority"] == "high"
    assert legend["chromatic_pixel_fraction_beyond_key_zone"] >= 0.005
    assert "question" in questions["legend_data_clearance"]
    assert "verdict" not in questions
    assert "blocking" not in questions


def test_layout_review_question_can_remain_routine_without_deciding_pixels(
    tmp_path: Path,
):
    image_path = tmp_path / "clear.png"
    image = Image.new("RGB", (1000, 600), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((620, 130, 650, 250), fill=(20, 90, 210))
    image.save(image_path)
    ocr = {
        "available": True,
        "words": [
            {"text": "Effect", "left": 300, "top": 20, "width": 100, "height": 30},
            {"text": "Analysis", "left": 420, "top": 20, "width": 120, "height": 30},
            {"text": "Treatment", "left": 700, "top": 120, "width": 130, "height": 28},
            {"text": "Control", "left": 700, "top": 160, "width": 115, "height": 28},
            {
                "text": "Observations",
                "left": 700,
                "top": 200,
                "width": 165,
                "height": 28,
            },
            {"text": "Group", "left": 700, "top": 240, "width": 95, "height": 28},
        ],
    }

    questions = _figure_layout_review_questions(
        image_path,
        ocr,
        width=1000,
        height=600,
    )

    assert questions["top_text_clearance"]["candidate_overlap_count"] == 0
    assert questions["top_text_clearance"]["priority"] == "routine"
    assert questions["legend_data_clearance"]["candidate"]["priority"] == "routine"
    assert questions["annotation_data_clearance"]["priority"] == "routine"
    assert questions["top_text_clearance"]["required"] is True
    assert questions["legend_data_clearance"]["required"] is True
    assert questions["annotation_data_clearance"]["required"] is True


def test_colored_mark_crossing_annotation_is_a_deterministic_candidate(
    tmp_path: Path, monkeypatch
):
    report, computation = _fixture(tmp_path)
    figure = Path(report.displays[0].artifact_path)
    image = Image.open(figure).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((345, 160, 355, 215), fill=(220, 90, 45))
    image.save(figure)
    computation.artifacts[0].sha256 = sha256_file(figure)
    ocr = {
        "available": True,
        "text": "Estimate control interval Hedges 4.071",
        "words": [
            {
                "text": "Estimate",
                "confidence": 96,
                "left": 80,
                "top": 30,
                "width": 90,
                "height": 20,
            },
            {
                "text": "Control",
                "confidence": 96,
                "left": 190,
                "top": 30,
                "width": 80,
                "height": 20,
            },
            {
                "text": "Interval",
                "confidence": 96,
                "left": 290,
                "top": 30,
                "width": 85,
                "height": 20,
            },
            {
                "text": "Hedges",
                "confidence": 96,
                "left": 395,
                "top": 30,
                "width": 80,
                "height": 20,
            },
            {
                "text": "4.071",
                "confidence": 82,
                "left": 300,
                "top": 155,
                "width": 110,
                "height": 60,
            },
        ],
    }

    candidates = figure_annotation_overlap_candidates(
        figure, ocr, width=640, height=400
    )
    assert candidates[0]["text"] == "4.071"
    assert candidates[0]["height_vs_median"] == 3.0
    assert candidates[0]["chromatic_pixel_fraction"] >= 0.02
    assert len(candidates[0]["box_fraction"]) == 4

    ocr["words"][-1]["confidence"] = 73
    assert (
        figure_annotation_overlap_candidates(figure, ocr, width=640, height=400) == []
    )
    ocr["words"][-1]["confidence"] = 82

    monkeypatch.setattr(
        "scientific_agent.linting.extract_figure_ocr", lambda _path: ocr
    )
    validation = validate_report(report, computation=computation)
    assert "figure_annotation_data_overlap" in {
        finding.code for finding in validation.findings
    }


def test_decompression_bomb_is_reported_as_invalid_figure(tmp_path, monkeypatch):
    image = tmp_path / "compressed-large-pixel-count.png"
    Image.new("L", (32, 32), color=0).save(image)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(ValueError, match="not a readable raster image"):
        inspect_figure(image)


def test_unregistered_final_display_artifact_is_blocking(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    report.displays.pop()

    validation = validate_report(report, computation=computation)

    assert "unregistered_report_artifact" in {
        finding.code for finding in validation.findings
    }


def test_unregistered_displays_are_still_sent_to_first_visual_audit(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    report.displays.clear()

    images, inputs = prepare_display_audit(report, computation)

    assert len(images) == 1
    assert {item["kind"] for item in inputs} == {"figure", "table"}
    assert all(item["registered"] is False for item in inputs)
    assert all(item["display_id"].startswith("unregistered:") for item in inputs)
    figure_input = next(item for item in inputs if item["kind"] == "figure")
    assert "ocr" in figure_input
    assert (
        figure_input["layout_review_questions"]["pixel_interpretation_authority"]
        == "Gemma"
    )
    assert next(item for item in inputs if item["kind"] == "table")["rows"] == [
        ["A|B", "1.25"],
        ["control", "0.00"],
    ]


def test_rendered_ocr_blocks_mixed_scales_and_zero_rounded_p_value(
    tmp_path: Path, monkeypatch
):
    report, computation = _fixture(tmp_path)
    results = tmp_path / "computations" / "exec-001" / "output" / "results.json"
    results.write_text('{"p_value":2.971749478841818e-13}\n', encoding="utf-8")
    computation.artifacts.append(
        ArtifactRef(
            path=str(results.resolve()),
            sha256=sha256_file(results),
            description="sandbox-generated analysis artifact",
        )
    )
    monkeypatch.setattr(
        "scientific_agent.linting.extract_figure_ocr",
        lambda _path: {
            "available": True,
            "text": (
                "Mean Difference 95% CI Hedges g "
                "Effect Size (Treatment minus Control) p = 0.000000"
            ),
            "words": [],
            "truncated": False,
        },
    )

    validation = validate_report(report, computation=computation)
    codes = {finding.code for finding in validation.findings}

    assert "figure_zero_rounded_nonzero_p_value" in codes
    assert "figure_mixed_incompatible_effect_scales" in codes


def test_rendered_ocr_blocks_bare_d_on_mean_difference_display(
    tmp_path: Path, monkeypatch
):
    report, computation = _fixture(tmp_path)
    monkeypatch.setattr(
        "scientific_agent.linting.extract_figure_ocr",
        lambda _path: {
            "available": True,
            "text": "Between-Group Contrast Mean Difference d = 5.00",
            "words": [],
            "truncated": False,
        },
    )

    validation = validate_report(report, computation=computation)
    codes = {finding.code for finding in validation.findings}

    assert "figure_ambiguous_bare_d_label" in codes


def test_rendered_ocr_allows_explicit_cohen_d_on_mean_difference_display(
    tmp_path: Path, monkeypatch
):
    report, computation = _fixture(tmp_path)
    monkeypatch.setattr(
        "scientific_agent.linting.extract_figure_ocr",
        lambda _path: {
            "available": True,
            "text": "Mean Difference; Cohen d = 3.45",
            "words": [],
            "truncated": False,
        },
    )

    validation = validate_report(report, computation=computation)
    codes = {finding.code for finding in validation.findings}

    assert "figure_ambiguous_bare_d_label" not in codes


def test_later_display_artifact_supersedes_same_logical_output(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    corrected_dir = (
        tmp_path / "computations" / "attempt-1" / "exec-002" / "output" / "figures"
    )
    corrected_dir.mkdir(parents=True)
    corrected = corrected_dir / "effect.png"
    Image.new("RGB", (800, 500), color=(220, 238, 246)).save(corrected, dpi=(300, 300))
    corrected_artifact = ArtifactRef(
        path=str(corrected.resolve()),
        sha256=sha256_file(corrected),
        description="corrected sandbox-generated display artifact",
    )
    computation.artifacts.append(corrected_artifact)
    report.displays[0].artifact_path = str(corrected.resolve())

    validation = validate_report(report, computation=computation)

    assert "unregistered_report_artifact" not in {
        finding.code for finding in validation.findings
    }


def test_model_supplied_display_numbers_are_rejected(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    report.displays[0].caption = "Figure 9. A misleading model-selected number."

    validation = validate_report(report, computation=computation)

    assert "model_supplied_display_number" in {
        finding.code for finding in validation.findings
    }


def test_reported_low_figure_dpi_is_blocking(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    figure = Path(report.displays[0].artifact_path)
    Image.new("RGB", (640, 400), color=(232, 244, 242)).save(figure, dpi=(200, 200))
    figure_hash = sha256_file(figure)
    computation.artifacts[0] = computation.artifacts[0].model_copy(
        update={"sha256": figure_hash}
    )

    validation = validate_report(report, computation=computation)

    assert "figure_dpi_below_minimum" in {
        finding.code for finding in validation.findings
    }

    Image.new("RGB", (640, 400), color=(232, 244, 242)).save(figure, dpi=(300, 300))
    computation.artifacts[0] = computation.artifacts[0].model_copy(
        update={"sha256": sha256_file(figure)}
    )
    validation = validate_report(report, computation=computation)
    assert "figure_dpi_below_minimum" not in {
        finding.code for finding in validation.findings
    }


def test_extreme_internal_figure_whitespace_is_blocking(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    figure = Path(report.displays[0].artifact_path)
    image = Image.new("RGB", (600, 1800), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 20, 550, 90), fill="black")
    draw.rectangle((40, 1450, 560, 1750), outline="black", width=12)
    image.save(figure, dpi=(300, 300))
    computation.artifacts[0] = computation.artifacts[0].model_copy(
        update={"sha256": sha256_file(figure)}
    )

    validation = validate_report(report, computation=computation)

    assert "figure_excessive_internal_whitespace" in {
        finding.code for finding in validation.findings
    }


def test_figure_alt_text_cannot_claim_nonnull_interval_crosses_zero(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    figure = report.displays[0]
    figure.alt_text = (
        "Effect plot with a 95% confidence interval crossing the zero reference line."
    )
    _add_machine_result(
        computation,
        tmp_path / "computations" / "exec-001" / "output" / "analysis.json",
        '{"primary":{"ci_95_lower":4.071144,"ci_95_upper":5.928856}}\n',
    )

    validation = validate_report(report, computation=computation)

    assert "figure_caption_interval_null_contradiction" in {
        finding.code for finding in validation.findings
    }

    figure.alt_text = (
        "Effect plot with a 95% confidence interval that does not cross zero."
    )
    validation = validate_report(report, computation=computation)
    assert "figure_caption_interval_null_contradiction" not in {
        finding.code for finding in validation.findings
    }


def test_reader_table_rejects_raw_computational_precision(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    table = Path(report.displays[1].artifact_path)
    table.write_text(
        "group,estimate\nA,1.23456789012\ncontrol,0.00000000000\n",
        encoding="utf-8",
    )
    computation.artifacts[1] = computation.artifacts[1].model_copy(
        update={"sha256": sha256_file(table)}
    )

    validation = validate_report(report, computation=computation)

    precision_finding = next(
        finding
        for finding in validation.findings
        if finding.code == "table_excessive_precision"
    )
    assert "four significant digits (not four decimal places)" in (
        precision_finding.message
    )
    assert "10.897" in precision_finding.message

    table.write_text("group,estimate\nA,1.2345\ncontrol,0.000\n", encoding="utf-8")
    computation.artifacts[1] = computation.artifacts[1].model_copy(
        update={"sha256": sha256_file(table)}
    )
    assert "table_excessive_precision" in {
        finding.code
        for finding in validate_report(report, computation=computation).findings
    }

    table.write_text("group,estimate\nA,1.235\ncontrol,0.000\n", encoding="utf-8")
    computation.artifacts[1] = computation.artifacts[1].model_copy(
        update={"sha256": sha256_file(table)}
    )
    assert "table_excessive_precision" not in {
        finding.code
        for finding in validate_report(report, computation=computation).findings
    }


def _add_machine_result(
    computation: ComputationEvidence,
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    computation.artifacts.append(
        ArtifactRef(
            path=str(path.resolve()),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
    )


def test_reader_table_rejects_zero_for_nonzero_machine_p_value(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    table = Path(report.displays[1].artifact_path)
    table.write_text("metric,value\np_value,0.0\n", encoding="utf-8")
    computation.artifacts[1] = computation.artifacts[1].model_copy(
        update={"sha256": sha256_file(table)}
    )
    _add_machine_result(
        computation,
        tmp_path / "computations" / "exec-001" / "output" / "results.json",
        '{"p_value": 2.9717494788418e-13}\n',
    )

    validation = validate_report(report, computation=computation)

    finding = next(
        item
        for item in validation.findings
        if item.code == "table_machine_result_contradiction"
    )
    assert finding.blocking
    assert "p_value=0.0" in finding.message
    assert "2.9717494788418E-13" in finding.message


def test_reader_table_accepts_four_digit_rounding_and_scientific_notation(
    tmp_path: Path,
):
    report, computation = _fixture(tmp_path)
    table = Path(report.displays[1].artifact_path)
    table.write_text(
        "metric,value\np value,2.972e-13\nmean_difference,5.000\nci_lower,4.071\n",
        encoding="utf-8",
    )
    computation.artifacts[1] = computation.artifacts[1].model_copy(
        update={"sha256": sha256_file(table)}
    )
    _add_machine_result(
        computation,
        tmp_path / "computations" / "exec-001" / "output" / "analysis.json",
        (
            '{"p_value": 2.9717494788418e-13, "mean_difference": 5.0, '
            '"confidence_interval": {"ci_lower": 4.071144254485707}}\n'
        ),
    )

    codes = {
        item.code for item in validate_report(report, computation=computation).findings
    }

    assert "table_machine_result_contradiction" not in codes
    assert "table_excessive_precision" not in codes


def test_reader_table_does_not_match_unrelated_numeric_field(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    table = Path(report.displays[1].artifact_path)
    table.write_text("metric,value\np_value,0.0\n", encoding="utf-8")
    computation.artifacts[1] = computation.artifacts[1].model_copy(
        update={"sha256": sha256_file(table)}
    )
    _add_machine_result(
        computation,
        tmp_path / "computations" / "exec-001" / "output" / "secondary.json",
        '{"secondary_p_value": 2.9717494788418e-13}\n',
    )

    codes = {
        item.code for item in validate_report(report, computation=computation).findings
    }

    assert "table_machine_result_contradiction" not in codes


def test_reader_wide_table_does_not_treat_group_header_as_metric(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    table = Path(report.displays[1].artifact_path)
    table.write_text(
        (
            "Metric,Treatment,Control,Difference\n"
            "N,20,20,\n"
            "Baseline mean,19.5,19.5,\n"
            "Change mean,5.000,0.000,5.000\n"
            "Welch t statistic,,,10.9\n"
        ),
        encoding="utf-8",
    )
    computation.artifacts[1] = computation.artifacts[1].model_copy(
        update={"sha256": sha256_file(table)}
    )
    _add_machine_result(
        computation,
        tmp_path / "computations" / "exec-001" / "output" / "analysis.json",
        (
            '{"group_counts": {"treatment": 20, "control": 20}, '
            '"site_counts": {"treatment": 10, "control": 10}, '
            '"welch_t_statistic": 10.897247358851683}\n'
        ),
    )

    codes = {
        item.code for item in validate_report(report, computation=computation).findings
    }

    assert "table_machine_result_contradiction" not in codes


def test_reader_wide_table_checks_explicit_group_metric_identity(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    table = Path(report.displays[1].artifact_path)
    table.write_text(
        "Metric,Treatment,Control\nBaseline mean,18.5,19.5\n",
        encoding="utf-8",
    )
    computation.artifacts[1] = computation.artifacts[1].model_copy(
        update={"sha256": sha256_file(table)}
    )
    _add_machine_result(
        computation,
        tmp_path / "computations" / "exec-001" / "output" / "analysis.json",
        (
            '{"groups": {"treatment": {"baseline_mean": 19.5}, '
            '"control": {"baseline_mean": 19.5}}}\n'
        ),
    )

    validation = validate_report(report, computation=computation)

    finding = next(
        item
        for item in validation.findings
        if item.code == "table_machine_result_contradiction"
    )
    assert "Baseline mean / Treatment=18.5" in finding.message
    assert "19.5" in finding.message


def test_latest_machine_result_supersedes_same_logical_repair_output(tmp_path: Path):
    report, computation = _fixture(tmp_path)
    table = Path(report.displays[1].artifact_path)
    table.write_text("metric,value\np_value,2.972e-13\n", encoding="utf-8")
    computation.artifacts[1] = computation.artifacts[1].model_copy(
        update={"sha256": sha256_file(table)}
    )
    _add_machine_result(
        computation,
        tmp_path
        / "computations"
        / "attempt-0"
        / "exec-001"
        / "output"
        / "results"
        / "analysis.json",
        '{"p_value": 0.0}\n',
    )
    _add_machine_result(
        computation,
        tmp_path
        / "computations"
        / "attempt-1"
        / "exec-001"
        / "output"
        / "results"
        / "analysis.json",
        '{"p_value": 2.9717494788418e-13}\n',
    )

    codes = {
        item.code for item in validate_report(report, computation=computation).findings
    }

    assert "table_machine_result_contradiction" not in codes


@pytest.mark.parametrize(
    "warning",
    [
        (
            "Figure includes Axes that are not compatible with tight_layout, "
            "so results might be incorrect.\n"
        ),
        (
            "UserWarning: Tight layout not applied. The bottom and top margins "
            "cannot be made large enough to accommodate all Axes decorations.\n"
        ),
        (
            "UserWarning: constrained_layout not applied because axes sizes "
            "collapsed to zero.\n"
        ),
    ],
)
def test_plotting_engine_layout_warning_is_blocking(tmp_path: Path, warning: str):
    report, computation = _fixture(tmp_path)
    stderr = tmp_path / "stderr.txt"
    stderr.write_text(warning, encoding="utf-8")
    computation.records = [
        ComputationRecord(
            execution_id="exec-001",
            language="python",
            code_sha256="abc",
            started_at="2026-07-14T12:00:00Z",
            duration_seconds=1.0,
            exit_code=0,
            status="succeeded",
            stdout_path=str(tmp_path / "stdout.txt"),
            stderr_path=str(stderr),
            artifacts=list(computation.artifacts),
        )
    ]

    validation = validate_report(report, computation=computation)

    assert "figure_render_warning" in {finding.code for finding in validation.findings}
