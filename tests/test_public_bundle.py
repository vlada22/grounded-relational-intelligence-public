import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_safety_validator_passes() -> None:
    subprocess.run([sys.executable, "scripts/validate_public_bundle.py"], cwd=ROOT, check=True)


def test_public_results_are_three_reviewed_backbones() -> None:
    data = json.loads((ROOT / "demo/data/results.json").read_text(encoding="utf-8"))
    assert set(data["models"]) == {"DINOv2", "DINOv3", "SigLIP 2"}
    assert data["models"]["DINOv2"]["license"] == "Apache-2.0"
    assert data["models"]["DINOv3"]["license"] == "DINOv3 License"
    assert data["models"]["SigLIP 2"]["license"] == "Apache-2.0"
    assert data["models"]["DINOv3"]["grid"] == [28, 28]
    assert data["models"]["DINOv2"]["grid"] == [32, 32]
    assert data["models"]["SigLIP 2"]["grid"] == [32, 32]


def test_single_canonical_article_and_static_demo() -> None:
    assert (ROOT / "ARTICLE.md").is_file()
    assert (ROOT / "PUBLICATION_SOURCE.json").is_file()
    assert not list(ROOT.glob("*linkedin*.md"))
    assert not list(ROOT.glob("*medium*.md"))
    assert (ROOT / "demo/index.html").is_file()
    assert (ROOT / "demo/app.js").is_file()
    assert (ROOT / "demo/styles.css").is_file()


def test_public_figures_exist() -> None:
    expected = {
        "dinov3-tower-a-similarity-depth.svg",
        "source-patch-grids.svg",
        "dinov2-perturbation-stability.svg",
        "dinov3-block5-cluster-evaluation.svg",
        "three-model-typed-relationship-f1.svg",
        "realistic-scene-cluster-structure.svg",
    }
    assert expected <= {path.name for path in (ROOT / "assets/figures").glob("*.svg")}
