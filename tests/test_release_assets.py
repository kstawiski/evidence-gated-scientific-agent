import json
import re
import tomllib
from pathlib import Path

import yaml

from scientific_agent.execution import R_ANALYSIS_BASELINE_PACKAGES


def test_public_release_assets_share_the_project_version():
    project_version = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    package = json.loads(Path("package.json").read_text(encoding="utf-8"))
    package_lock = json.loads(Path("package-lock.json").read_text(encoding="utf-8"))
    citation = Path("CITATION.cff").read_text(encoding="utf-8")
    init = Path("scientific_agent/__init__.py").read_text(encoding="utf-8")

    assert package["version"] == project_version
    assert package_lock["version"] == project_version
    assert package_lock["packages"][""]["version"] == project_version
    assert re.search(r"^version:\s*([^\s]+)", citation, re.M).group(1) == (
        project_version
    )
    assert re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.M).group(1) == (
        project_version
    )

    compose = yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))
    image_tags = {
        service["image"].rsplit(":", 1)[1]
        for service in compose["services"].values()
        if service.get("image", "").startswith("evidence-bench")
    }
    worker_image = compose["services"]["environment-worker"]["environment"][
        "SCIENTIFIC_AGENT_WORKER_IMAGE_ID"
    ]
    image_tags.add(worker_image.rsplit(":", 1)[1])
    assert image_tags == {project_version}


def test_container_smoke_derives_its_image_tag_from_project_metadata():
    smoke = Path("scripts/container_boundary_smoke.sh").read_text(encoding="utf-8")

    assert "tomllib" in smoke
    assert "evidence-bench-browser:${version}" in smoke
    assert not re.search(r"evidence-bench(?:-browser|-packages)?:\d+\.\d+\.\d+", smoke)


def test_container_smoke_is_isolated_from_the_production_compose_project():
    smoke = Path("scripts/container_boundary_smoke.sh").read_text(encoding="utf-8")

    assert "evidence-bench-smoke-$$" in smoke
    assert 'if [ "$project_name" = evidence-bench ]' in smoke
    assert "BROWSER_BIND_ADDRESS=127.0.0.1" in smoke
    assert "mktemp -d" in smoke


def test_wheel_and_container_include_webui_integration_sources():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    forced = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert forced["skills/evidence-bench"].endswith("release_assets/evidence-bench")
    assert forced["integrations/a2a"].endswith("release_assets/a2a")
    assert "COPY skills/evidence-bench ./skills/evidence-bench" in dockerfile
    assert "COPY integrations/a2a ./integrations/a2a" in dockerfile


def test_runtime_image_has_no_setuid_bubblewrap_and_privilege_model_is_documented():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    threat_model = Path("docs/THREAT_MODEL.md").read_text(encoding="utf-8")

    assert "chmod u+s /usr/bin/bwrap" not in dockerfile
    assert "SYS_ADMIN" in threat_model
    assert "unconfined Docker\n  seccomp/AppArmor" in threat_model
    assert "does not install a setuid bubblewrap binary" in threat_model


def test_runtime_image_includes_publication_figure_r_baseline():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "fonts-open-sans" in dockerfile
    assert "patchwork_1.2.0.tar.gz" in dockerfile
    for package in R_ANALYSIS_BASELINE_PACKAGES:
        apt_package = (
            "r-bioc-complexheatmap"
            if package == "ComplexHeatmap"
            else f"r-cran-{package.lower()}"
        )
        assert apt_package in dockerfile
