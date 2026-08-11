# Grounded Relational Intelligence

Public, publication-focused distillation of Article 3 in the Grounded Visual Intelligence series.

> Can frozen Vision Transformer features be turned into an inspectable spatial representation before we train an explicit segmentation or scene-graph model?

- [Read **From Pixels to Relationships**](ARTICLE.md)
- [Open the **Transformer Spatial Explorer**](https://vlada22.github.io/grounded-relational-intelligence-public/)

The central result is deliberately simple: **frozen transformer features contain useful spatial structure, but no single layer wins every probe and similarity does not automatically compose into a reliable semantic relationship.**

## Experiment

The controlled comparison uses three frozen backbones:

- DINOv2 ViT-S/14
- DINOv3 ViT-S/16
- SigLIP 2 Base Patch/16 NaFlex

Each is inspected at blocks 2, 5, and 11 under the same deterministic probe across four controlled scene variants. The public evidence covers clean-reference retrieval, cluster alignment, tolerant boundary quality, grouping stability, relationship diagnostics, and one unlabeled realistic-scene stress test.

Headline observations:

- DINOv2 retrieval peaks at block 5 (`0.8375`), while clustering and boundary alignment are strongest at block 2.
- DINOv3 retrieval peaks at block 11 (`0.866667` P@12), clustering at block 5 (`0.562904` ARI), and boundary alignment at block 2 (`0.853504`).
- SigLIP 2 is strongest at block 2 for retrieval (`0.8125`), ARI (`0.669834`), and boundary F1 (`0.929080`).
- At grouping seed `17`, the strongest mean perturbation grouping stability occurs at DINOv2 block 11 (`0.888689`), DINOv3 block 2 (`0.875950`), and SigLIP 2 block 2 (`0.785439`).
- The compact relationship summary is weak for all three backbones: diagnostic macro F1 remains below `0.20`, and the fixed embedding-cosine diagnostic is `0.0` on every selected run.

These measurements describe this controlled protocol; they are not a model leaderboard.

## Public boundary

This repository intentionally contains only the material needed for publication, inspection, and deterministic verification:

- canonical article text;
- aggregate reviewed measurements;
- publication figures;
- the static public explorer;
- public validation/tests;
- model identifiers, revisions, and license references.

It does **not** contain raw feature tensors, foundation-model weights, gated-model runtime bundles, or the private research handoffs. The originating private checkpoint is recorded in [`PUBLICATION_SOURCE.json`](PUBLICATION_SOURCE.json).

## Methodology boundaries

- Reported retrieval P@k uses five predeclared truth regions and the highest-purity patch in each at evaluation time.
- Exact masks are projected to patch-level evaluation truth only after candidate construction.
- PCA-8 and `k=6` are fixed probe settings; the originally planned `k` sweep was not executed, so `k=6` is not claimed as optimal.
- Headline grouping stability uses predeclared seed `17`.
- Relationship-table layer selection is post-hoc best-observed base-scene diagnostic macro F1, not held-out model selection.
- `embedding_similar` is a predicted cosine rule. Its diagnostic target combines the declared `same_structure` and `texture_similar` pairs; it is not itself a semantic ground-truth relation type.
- The realistic scene has no labels and supports descriptive probe behaviour only.

## Run the explorer locally

```bash
python -m http.server 8000
```

Open `http://localhost:8000/demo/`.

The explorer is static and performs no foundation-model inference. It uses the aggregate reviewed values in `demo/data/results.json` and preserves the correct 32×32 or 28×28 patch geometry for the selected backbone.

## Validate the public bundle

Requires Python 3.11 or newer.

```bash
python scripts/validate_public_bundle.py
python -m pytest
```

## Repository layout

```text
ARTICLE.md                     canonical Article 03 text
PUBLICATION_SOURCE.json        private-source checkpoint and distillation record
assets/controlled-scene.svg   public controlled scene
assets/figures/                publication SVGs
demo/                          static three-model evidence explorer
scripts/validate_public_bundle.py
scripts/rebuild_figures.py     aggregate figure helper
THIRD_PARTY.md                 model references and license boundaries
```

## Model references

Exact identifiers and upstream license references are documented in [THIRD_PARTY.md](THIRD_PARTY.md).
