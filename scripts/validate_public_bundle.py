from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {"DINOv2", "DINOv3", "SigLIP 2"}
EXPECTED_FIGURES = {
    "dinov3-tower-a-similarity-depth.svg",
    "source-patch-grids.svg",
    "dinov2-perturbation-stability.svg",
    "dinov3-block5-cluster-evaluation.svg",
    "three-model-typed-relationship-f1.svg",
    "realistic-scene-cluster-structure.svg",
}
EXPECTED_DEMO_FILES = {"index.html", "styles.css", "app.js"}
UNEXPECTED_BINARY_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".onnx", ".npy", ".npz"}


def main() -> None:
    violations: list[str] = []

    data = json.loads((ROOT / "demo/data/results.json").read_text(encoding="utf-8"))
    if set(data["models"]) != EXPECTED_MODELS:
        violations.append("results.json must contain exactly DINOv2, DINOv3 and SigLIP 2")
    if data["models"].get("DINOv3", {}).get("license") != "DINOv3 License":
        violations.append("DINOv3 must retain its recorded upstream license boundary")
    for name in ("DINOv2", "SigLIP 2"):
        if data["models"].get(name, {}).get("license") != "Apache-2.0":
            violations.append(f"{name} public metadata must retain Apache-2.0")
    if data["models"].get("DINOv3", {}).get("grid") != [28, 28]:
        violations.append("DINOv3 must retain its reviewed 28x28 patch grid")
    for name in ("DINOv2", "SigLIP 2"):
        if data["models"].get(name, {}).get("grid") != [32, 32]:
            violations.append(f"{name} must retain its reviewed 32x32 patch grid")

    demo_files = {path.name for path in (ROOT / "demo").iterdir() if path.is_file()}
    if not EXPECTED_DEMO_FILES <= demo_files:
        violations.append("demo is missing one or more required static files")

    figure_files = {path.name for path in (ROOT / "assets/figures").glob("*.svg")}
    if not EXPECTED_FIGURES <= figure_files:
        violations.append("final publication figure set is incomplete")

    if not (ROOT / "ARTICLE.md").is_file():
        violations.append("canonical ARTICLE.md is missing")
    if not (ROOT / "PUBLICATION_SOURCE.json").is_file():
        violations.append("PUBLICATION_SOURCE.json is missing")
    article_variants = [path for path in ROOT.glob("*article*.md") if path.name.lower() != "article.md"]
    if article_variants:
        violations.append(f"duplicate article variants found: {article_variants}")

    forbidden_names = {"patch-features.npy", "patch-features.npz"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in UNEXPECTED_BINARY_SUFFIXES or path.name in forbidden_names:
            violations.append(f"unexpected feature/model artifact: {path.relative_to(ROOT)}")

    if violations:
        raise SystemExit("PUBLIC BUNDLE VALIDATION FAILED\n" + "\n".join(f"- {item}" for item in violations))
    print("public bundle validation passed")


if __name__ == "__main__":
    main()
