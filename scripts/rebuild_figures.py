from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "demo/data/results.json").read_text(encoding="utf-8"))
OUT = ROOT / "assets/figures"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"DINOv2": "#62dfc0", "DINOv3": "#8fb5ff", "SigLIP 2": "#efc56d"}
MODELS = ("DINOv2", "DINOv3", "SigLIP 2")
LAYERS = (2, 5, 11)


def header(title: str, subtitle: str, width: int = 1000, height: int = 560) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#071217"/>',
        f'<text x="54" y="58" fill="#e7f2ee" font-family="Arial" font-size="28" font-weight="700">{title}</text>',
        f'<text x="54" y="88" fill="#91a7a7" font-family="Arial" font-size="15">{subtitle}</text>',
    ]


def line_chart(filename: str, metric: str, title: str, subtitle: str) -> None:
    x0, y0, x1, y1 = 90, 455, 930, 125
    out = header(title, subtitle)
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = y0 - (y0 - y1) * tick
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#20333a"/>')
        out.append(f'<text x="48" y="{y + 5:.1f}" fill="#91a7a7" font-family="Arial" font-size="12">{tick:.2f}</text>')
    for index, layer in enumerate(LAYERS):
        x = x0 + (x1 - x0) * index / 2
        out.append(f'<text x="{x - 10:.1f}" y="490" fill="#91a7a7" font-family="Arial" font-size="13">L{layer}</text>')
    for model in MODELS:
        points = []
        for index, layer in enumerate(LAYERS):
            value = RESULTS["models"][model]["layers"][str(layer)][metric]
            x = x0 + (x1 - x0) * index / 2
            y = y0 - (y0 - y1) * value
            points.append((x, y, value))
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        out.append(f'<polyline fill="none" stroke="{COLORS[model]}" stroke-width="4" points="{coords}"/>')
        for x, y, value in points:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{COLORS[model]}"/>')
            out.append(f'<text x="{x + 10:.1f}" y="{y - 10:.1f}" fill="{COLORS[model]}" font-family="Arial" font-size="12">{value:.3f}</text>')
    legend_x = 610
    for index, model in enumerate(MODELS):
        x = legend_x + index * 120
        out.append(f'<circle cx="{x}" cy="52" r="6" fill="{COLORS[model]}"/>')
        out.append(f'<text x="{x + 14}" y="57" fill="#e7f2ee" font-family="Arial" font-size="13">{model}</text>')
    out.append("</svg>")
    (OUT / filename).write_text("\n".join(out), encoding="utf-8")


def relationship_chart() -> None:
    out = header(
        "Best-observed base-scene relationship diagnostics",
        "Post-hoc descriptive layer per model; embedding similarity is a diagnostic target, not a semantic truth type.",
    )
    x0, y0, maximum, group_gap, bar_width = 90, 455, 0.5, 285, 46
    for tick in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        y = y0 - (tick / maximum) * 300
        out.append(f'<line x1="70" y1="{y:.1f}" x2="940" y2="{y:.1f}" stroke="#20333a"/>')
        out.append(f'<text x="36" y="{y + 4:.1f}" fill="#91a7a7" font-family="Arial" font-size="12">{tick:.1f}</text>')
    for group, model in enumerate(MODELS):
        record = RESULTS["models"][model]["typed_relationship_selected"]
        values = (
            ("Adjacent", record["adjacent_f1"], "#62dfc0"),
            ("Near", record["near_f1"], "#9b8cff"),
            ("Embedding", record["embedding_similarity_diagnostic_f1"], "#efc56d"),
        )
        group_x = x0 + group * group_gap
        for index, (label, value, colour) in enumerate(values):
            x = group_x + index * (bar_width + 24)
            height = (value / maximum) * 300
            y = y0 - height
            out.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{height:.1f}" rx="5" fill="{colour}"/>')
            out.append(f'<text x="{x + bar_width / 2}" y="{y - 10:.1f}" text-anchor="middle" fill="#e7f2ee" font-family="Arial" font-size="11">{value:.3f}</text>')
            out.append(f'<text x="{x + bar_width / 2}" y="480" text-anchor="middle" fill="#91a7a7" font-family="Arial" font-size="10">{label}</text>')
        out.append(f'<text x="{group_x + 94}" y="520" text-anchor="middle" fill="{COLORS[model]}" font-family="Arial" font-size="15" font-weight="700">{model} · L{record["layer"]}</text>')
    out.append("</svg>")
    (OUT / "typed-relationships.svg").write_text("\n".join(out), encoding="utf-8")


def realistic_chart() -> None:
    out = header(
        "Realistic clutter keeps the fixed probe depth-sensitive",
        "Connected candidate regions. Descriptive only — the realistic scene has no semantic ground truth.",
    )
    x0, y0, x1, maximum = 110, 450, 910, 30
    for tick in (0, 5, 10, 15, 20, 25, 30):
        y = y0 - (tick / maximum) * 300
        out.append(f'<line x1="80" y1="{y:.1f}" x2="930" y2="{y:.1f}" stroke="#20333a"/>')
        out.append(f'<text x="46" y="{y + 4:.1f}" fill="#91a7a7" font-family="Arial" font-size="12">{tick}</text>')
    for model in MODELS:
        points = []
        for index, layer in enumerate(LAYERS):
            value = RESULTS["models"][model]["realistic_candidate_regions"][str(layer)]
            x = x0 + (x1 - x0) * index / 2
            y = y0 - (value / maximum) * 300
            points.append((x, y, value))
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        out.append(f'<polyline fill="none" stroke="{COLORS[model]}" stroke-width="4" points="{coords}"/>')
        for x, y, value in points:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{COLORS[model]}"/>')
            out.append(f'<text x="{x + 10:.1f}" y="{y - 10:.1f}" fill="{COLORS[model]}" font-family="Arial" font-size="12">{value}</text>')
    for index, layer in enumerate(LAYERS):
        x = x0 + (x1 - x0) * index / 2
        out.append(f'<text x="{x}" y="490" text-anchor="middle" fill="#91a7a7" font-family="Arial" font-size="13">L{layer}</text>')
    out.append("</svg>")
    (OUT / "realistic-candidate-regions.svg").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    line_chart("retrieval-by-depth.svg", "retrieval", "Same-region retrieval changes with depth", "Controlled base scene; coverage matched across model patch grids.")
    line_chart("clustering-by-depth.svg", "ari", "Region alignment is strongest at different depths", "Mean ARI across seeds 7, 17 and 29 against patch-level evaluation truth.")
    relationship_chart()
    realistic_chart()
    print(f"rebuilt aggregate public figures in {OUT}")
