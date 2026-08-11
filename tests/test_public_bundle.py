import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_safety_validator_passes() -> None:
    subprocess.run([sys.executable, "scripts/validate_public_bundle.py"], cwd=ROOT, check=True)


def test_public_results_are_two_open_backbones() -> None:
    data = json.loads((ROOT / "demo/data/results.json").read_text(encoding="utf-8"))
    assert set(data["models"]) == {"DINOv2", "SigLIP 2"}
    assert {model["license"] for model in data["models"].values()} == {"Apache-2.0"}


def test_single_canonical_article_and_static_demo() -> None:
    assert (ROOT / "ARTICLE.md").is_file()
    assert not list(ROOT.glob("*linkedin*.md"))
    assert not list(ROOT.glob("*medium*.md"))
    assert (ROOT / "demo/index.html").is_file()
    assert (ROOT / "demo/app.js").is_file()
    assert (ROOT / "demo/styles.css").is_file()


def test_public_figures_exist() -> None:
    expected = {
        "retrieval-by-depth.svg",
        "clustering-by-depth.svg",
        "typed-relationships.svg",
        "realistic-candidate-regions.svg",
    }
    assert expected <= {path.name for path in (ROOT / "assets/figures").glob("*.svg")}
