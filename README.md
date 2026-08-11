# Grounded Relational Intelligence — Public Evidence Edition

A compact public companion to **Article 03: From Pixels to Relationships**.

The project asks a narrow question:

> How far can frozen transformer patch features take us from pixels to regions and typed spatial relationships before the application has to add structure explicitly?

- [Read the complete article](ARTICLE.md)
- [Open the interactive evidence explorer](https://vlada22.github.io/grounded-relational-intelligence-public/)

The central result is deliberately modest: **useful patch organization exists, but the best depth depends on the question, and feature similarity does not automatically compose into a reliable semantic relationship graph.**

## What is public here

This repository is intentionally smaller than the private research workspace. It contains only:

- a repository-authored procedural controlled scene;
- reviewed **aggregate** measurements from DINOv2 and SigLIP 2;
- publication SVGs regenerated from those measurements;
- a dependency-free static evidence explorer;
- deterministic validation and figure-building scripts;
- tests for the public-safety boundary;
- the single canonical Article 03 text.

It contains **no model weights, credentials, token workflows, raw full-dimensional feature archives, or gated/separately licensed research-model materials**.

## Public evidence

On the controlled base scene:

- DINOv2 retrieval peaks at block 5 (`0.8375`), while clustering and boundary alignment are strongest at block 2;
- DINOv2 grouping stability is strongest at block 11 (`0.8887`);
- SigLIP 2 is strongest at block 2 for retrieval (`0.8125`), ARI (`0.6698`), boundary F1 (`0.9291`), and grouping stability (`0.7854`);
- the selected typed-relationship runs remain weak, with macro F1 `0.1690` for DINOv2 and `0.1944` for SigLIP 2;
- the fixed `embedding_similar` edge rule recovers none of the declared semantic-similarity edges in those selected runs.

These are controlled-probe measurements, not a model leaderboard.

## Reproduce the public figures

Requires Python 3.11 or newer. The publication scripts use only the standard library.

```bash
python scripts/validate_public_bundle.py
python scripts/rebuild_figures.py
```

`rebuild_figures.py` reads `demo/data/results.json` and rewrites the SVGs in `assets/figures/`.

## Run the explorer locally

```bash
python -m http.server 8000
```

Open `http://localhost:8000/demo/`.

The explorer performs no model inference. It reads only the public aggregate evidence JSON and the repository-authored source image.

## Repository layout

```text
ARTICLE.md                     single canonical article
assets/controlled-scene.svg   repository-authored procedural scene
assets/figures/                deterministic publication SVGs
demo/                          static GitHub Pages explorer
scripts/rebuild_figures.py     regenerate figures from aggregate evidence
scripts/validate_public_bundle.py
THIRD_PARTY.md                 model/license boundary
```

## Public-safety boundary

The public validation script enforces the approved model set, checks that only aggregate evidence is exposed, and rejects model-weight formats, environment-secret files, credential assignments, and duplicate article variants.

For model licensing and redistribution boundaries, see [THIRD_PARTY.md](THIRD_PARTY.md).
