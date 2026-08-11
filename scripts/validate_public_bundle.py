from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {"DINOv2", "SigLIP 2"}
EXPECTED_FIGURES = {
    "retrieval-by-depth.svg",
    "clustering-by-depth.svg",
    "typed-relationships.svg",
    "realistic-candidate-regions.svg",
}
EXPECTED_DEMO_FILES = {"index.html", "styles.css", "app.js"}
UNEXPECTED_BINARY_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".onnx"}


def main() -> None:
    violations: list[str] = []

    data = json.loads((ROOT / "demo/data/results.json").read_text(encoding="utf-8"))
    if set(data["models"]) != EXPECTED_MODELS:
        violations.append("results.json must contain exactly the two experiment backbones")
    for model in data["models"].values():
        if model["license"] != "Apache-2.0":
            violations.append("published model metadata must match the recorded upstream license")

    demo_files = {path.name for path in (ROOT / "demo").iterdir() if path.is_file()}
    if not EXPECTED_DEMO_FILES <= demo_files:
        violations.append("demo is missing one or more required static files")

    figure_files = {path.name for path in (ROOT / "assets/figures").glob("*.svg")}
    if not EXPECTED_FIGURES <= figure_files:
        violations.append("publication figure set is incomplete")

    article_variants = [path for path in ROOT.glob("*article*.md") if path.name.lower() != "article.md"]
    if article_variants:
        violations.append(f"duplicate article variants found: {article_variants}")

    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in UNEXPECTED_BINARY_SUFFIXES:
            violations.append(f"unexpected binary artifact: {path.relative_to(ROOT)}")

    if violations:
        raise SystemExit("PUBLIC BUNDLE VALIDATION FAILED\n" + "\n".join(f"- {item}" for item in violations))
    print("public bundle validation passed")


if __name__ == "__main__":
    main()
