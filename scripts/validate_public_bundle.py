from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".toml", ".json", ".html", ".css", ".js", ".yml", ".yaml", ".txt", ".svg"}
FORBIDDEN_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".onnx"}
ALLOWED_MODELS = {"DINOv2", "SigLIP 2"}
SECRET_ASSIGNMENT = re.compile(r"(?i)(token|secret|api[_-]?key)\s*[:=]\s*['\"][^'\"]+")


def main() -> None:
    violations: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"model/binary artifact is not allowed: {relative}")
        if path.name == ".env" or path.name.startswith(".env."):
            violations.append(f"environment secret file is not allowed: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            if path != Path(__file__) and SECRET_ASSIGNMENT.search(text):
                violations.append(f"possible credential assignment in {relative}")

    data = json.loads((ROOT / "demo/data/results.json").read_text(encoding="utf-8"))
    if set(data["models"]) != ALLOWED_MODELS:
        violations.append("public results must contain exactly the two approved public backbones")
    for model in data["models"].values():
        if model["license"] != "Apache-2.0":
            violations.append("all public model evidence must report Apache-2.0 upstream")

    article_variants = [path for path in ROOT.glob("*article*.md") if path.name.lower() != "article.md"]
    if article_variants:
        violations.append(f"duplicate article variants found: {article_variants}")

    if violations:
        raise SystemExit("PUBLIC BUNDLE VALIDATION FAILED\n" + "\n".join(f"- {item}" for item in violations))
    print("public bundle validation passed")


if __name__ == "__main__":
    main()
