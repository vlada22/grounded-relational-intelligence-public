# Grounded Relational Intelligence

Experiments in turning frozen vision-transformer features into inspectable regions and typed spatial relationships.

The third article in the grounded visual intelligence series asks:

> How far can patch features take us from pixels to regions and explicit relationships before the application has to add structure of its own?

- [Read **From Pixels to Relationships**](ARTICLE.md)
- [Open the interactive evidence explorer](https://vlada22.github.io/grounded-relational-intelligence-public/)

The central result is simple: **patch features can organize a scene surprisingly well, but useful similarity is not the same thing as a reliable relationship.**

## Experiment

The controlled experiment uses a repository-authored procedural scene and two frozen backbones:

- DINOv2 ViT-S/14
- SigLIP 2 base patch16 NaFlex

Both are inspected at blocks 2, 5, and 11 under the same deterministic probe. The published evidence covers same-region retrieval, cluster alignment, boundary quality, grouping stability, typed relationship recovery, and a qualitative realistic-scene stress test.

A few headline results:

- DINOv2 retrieval peaks at block 5 (`0.8375`), while clustering and boundary alignment are strongest at block 2.
- DINOv2 grouping stability is strongest at block 11 (`0.8887`).
- SigLIP 2 is strongest at block 2 for retrieval (`0.8125`), ARI (`0.6698`), boundary F1 (`0.9291`), and grouping stability (`0.7854`).
- The selected typed-relationship runs remain difficult: macro F1 is `0.1690` for DINOv2 and `0.1944` for SigLIP 2.
- The fixed `embedding_similar` rule recovers none of the declared semantic-similarity edges in those selected runs.

These measurements are intended to expose representation trade-offs, not rank the two models.

## Rebuild the figures

Requires Python 3.11 or newer. The figure builder uses only the standard library.

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

The explorer puts the aggregate measurements next to the controlled scene so model, layer, metric, and source-patch coordinates can be inspected together.

## Repository layout

```text
ARTICLE.md                     canonical Article 03 text
assets/controlled-scene.svg   procedural controlled scene
assets/figures/                deterministic publication SVGs
demo/                          static evidence explorer
scripts/rebuild_figures.py     regenerate figures from aggregate evidence
scripts/validate_public_bundle.py
THIRD_PARTY.md                 model references
```

## Model references

The model identifiers and upstream references used for the published experiment are documented in [THIRD_PARTY.md](THIRD_PARTY.md).
