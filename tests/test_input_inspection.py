import json
import zipfile

from scientific_agent.input_inspection import build_input_profile


def test_delimited_intake_profiles_shape_types_and_missingness_without_values(tmp_path):
    (tmp_path / "cohort.csv").write_text(
        "patient_id,age,arm,outcome\nP001,61,A,4.5\nP002,,B,NA\nP003,57,A,7.0\n",
        encoding="utf-8",
    )

    profile = build_input_profile(tmp_path)

    assert profile.total_files == profile.profiled_files == 1
    source = profile.files[0]
    assert source.path == "/workspace/cohort.csv"
    assert source.detected_format == "delimited_text"
    assert source.rows_total == 3
    columns = {column.name: column for column in source.columns}
    assert columns["age"].missing_count == 1
    assert columns["age"].missing_fraction == 1 / 3
    assert columns["age"].inferred_types == ["integer"]
    assert columns["outcome"].missing_count == 1
    assert columns["arm"].candidate_role_labels == ["A", "B"]
    assert columns["arm"].candidate_role_labels_complete is True
    assert columns["patient_id"].candidate_role_labels == []
    serialized = json.dumps(profile.model_dump(mode="json"))
    assert "P001" not in serialized
    assert "4.5" not in serialized


def test_json_record_intake_counts_absent_keys_as_missing(tmp_path):
    (tmp_path / "records.json").write_text(
        json.dumps([{"sample": "S1", "value": 2}, {"sample": "S2"}]),
        encoding="utf-8",
    )

    profile = build_input_profile(tmp_path)

    source = profile.files[0]
    assert source.rows_total == 2
    columns = {column.name: column for column in source.columns}
    assert columns["value"].missing_count == 1
    assert columns["value"].non_missing_count == 1


def test_archive_intake_lists_members_without_extraction(tmp_path):
    archive = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("tables/result.csv", "x,y\n1,2\n")
        handle.writestr("../escape.txt", "not extracted")

    profile = build_input_profile(tmp_path)

    source = profile.files[0]
    assert source.detected_format == "zip"
    assert source.details["archive_extracted"] is False
    assert {item["path"] for item in source.details["members"]} == {
        "tables/result.csv",
        "../escape.txt",
    }
    assert not (tmp_path.parent / "escape.txt").exists()


def test_input_profile_reuses_controller_manifest_hash(tmp_path):
    source = tmp_path / "large.bin"
    source.write_bytes(b"large-input-fixture")

    profile = build_input_profile(tmp_path, {"large.bin": "a" * 64})

    assert profile.files[0].sha256 == "a" * 64


def test_xlsx_intake_profiles_first_sheet_missingness_without_cell_values(tmp_path):
    workbook = tmp_path / "cohort.xlsx"
    shared = """<?xml version="1.0" encoding="UTF-8"?>
    <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <si><t>patient_id</t></si><si><t>age</t></si><si><t>P001</t></si>
      <si><t>P002</t></si>
    </sst>"""
    sheet = """<?xml version="1.0" encoding="UTF-8"?>
    <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <sheetData>
        <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
        <row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>61</v></c></row>
        <row r="3"><c r="A3" t="s"><v>3</v></c></row>
      </sheetData>
    </worksheet>"""
    with zipfile.ZipFile(workbook, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)

    profile = build_input_profile(tmp_path)

    source = profile.files[0]
    columns = {column.name: column for column in source.columns}
    assert source.detected_format == "xlsx"
    assert source.rows_total == 2
    assert columns["age"].missing_count == 1
    assert columns["age"].inferred_types == ["integer"]
    serialized = json.dumps(profile.model_dump(mode="json"))
    assert "P001" not in serialized
    assert source.details["values_included_in_profile"] is False


def test_xlsx_intake_exposes_value_free_sheet_names_and_dimensions(tmp_path):
    workbook = tmp_path / "cohort.xlsx"
    workbook_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <sheets>
        <sheet name="training_rfs" sheetId="1" r:id="rId1"/>
        <sheet name="validation_pfs" sheetId="2" r:id="rId2"/>
      </sheets>
    </workbook>"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
      <Relationship Id="rId1" Target="worksheets/sheet1.xml"/>
      <Relationship Id="rId2" Target="worksheets/sheet2.xml"/>
    </Relationships>"""
    sheet1 = """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <dimension ref="A1:C2558"/><sheetData><row r="1"/></sheetData></worksheet>"""
    sheet2 = """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <dimension ref="A1:D323"/><sheetData><row r="1"/></sheetData></worksheet>"""
    with zipfile.ZipFile(workbook, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet1)
        archive.writestr("xl/worksheets/sheet2.xml", sheet2)

    source = build_input_profile(tmp_path).files[0]

    assert source.details["worksheet_structure"] == [
        {"name": "training_rfs", "rows_declared": 2558, "columns_declared": 3},
        {"name": "validation_pfs", "rows_declared": 323, "columns_declared": 4},
    ]


def test_xlsx_declared_dimensions_count_used_range_not_absolute_indices(tmp_path):
    workbook = tmp_path / "offset.xlsx"
    sheet = """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <dimension ref="C5:F10"/><sheetData><row r="5"/></sheetData></worksheet>"""
    with zipfile.ZipFile(workbook, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)

    structure = build_input_profile(tmp_path).files[0].details["worksheet_structure"]

    assert structure == [{"name": "sheet1", "rows_declared": 6, "columns_declared": 4}]
