# From Pixels to Relationships

## What I learned trying to turn transformer features into a spatial graph

## Artifacts

- [GitHub repository](https://github.com/vlada22/grounded-relational-intelligence-public)
- [Interactive evidence explorer](https://vlada22.github.io/grounded-relational-intelligence-public/)

The first two experiments in this series taught me to be suspicious of shortcuts.

In the first, the shortcut was language: a model could describe a video fluently, but measurable answers became much more trustworthy once observations were turned into explicit evidence and deterministic tools did the counting and timing.

In the second, the shortcut was visual plausibility: a depth map could look coherent while the geometry was still wrong enough to make object sizes and distances unreliable.

For the third experiment, I wanted to push the same idea one level deeper.

Modern vision transformers produce dense patch features that are remarkably useful. Nearby patches often organize together. Similar structures often become close in feature space. Intermediate layers can preserve boundaries that later layers smooth away.

That makes it tempting to say that the model has already discovered the *relationships* in a scene.

I wanted to test that temptation.

> If a frozen vision transformer gives us useful patch features, how far can we go from **pixels**, to **regions**, to an explicit **relationship graph** before we start inventing structure the model did not actually give us?

That question became **Grounded Relational Intelligence**.

The main result surprised me less because the models failed than because of *where* they failed:

> **Patch features can organize a scene surprisingly well. But useful similarity is not the same thing as a reliable relationship.**

That distinction became the thread running through the whole experiment.

![The controlled scene used to test feature similarity, clustering, and spatial relationships.](assets/controlled-scene.svg)

*Figure 1. A deliberately simple procedural scene with repeated structures, a texture trap, occlusion, containment, adjacency, and nearby objects. Ground-truth labels exist only for evaluation.*

## Why start with an artificial scene?

Because real images are too forgiving when the question is vague.

If I show a transformer a street scene and its features look coherent, it is very easy to convince myself that the representation is meaningful. But unless I know the scene exactly, I cannot tell whether a cluster boundary is correct, whether two similar regions should actually be related, or whether a visually convincing grouping is simply a texture shortcut.

So I built a 448 × 448 procedural scene with several deliberate traps:

- two towers with related structure but different appearance;
- a warehouse sharing a repeated texture pattern with one tower;
- a car partially hidden behind a foreground barrier;
- regions that are adjacent;
- regions that are merely near;
- and a door geometrically contained by the warehouse.

The scene comes with exact masks and a small relationship vocabulary. But those labels are kept out of feature extraction, PCA, clustering, and candidate graph construction. They enter only afterward, when the candidates are evaluated.

That separation matters. Otherwise it would be too easy to build the answer into the representation and then congratulate the representation for finding it.

## Two public backbones, one fixed probe

For the public experiment I use two openly licensed frozen backbones:

- **DINOv2 ViT-S/14**;
- **SigLIP 2 base patch16 NaFlex**.

Both are used only as feature extractors. No fine-tuning is involved.

I also ran an additional separately licensed backbone during the private research phase. It reinforced the same qualitative conclusion about depth sensitivity, but I intentionally keep that model, its access path, its raw features, and its derived interactive artifacts out of this public repository. The public evidence here stands on the two open backbones above.

For each model I inspect blocks 2, 5, and 11. The candidate-region probe is deliberately plain:

```text
frozen patch features
  -> L2 normalization
  -> per-image PCA to 8 dimensions
  -> seeded k-means, k = 6
  -> three spatial majority passes
  -> connected components
```

The point was not to optimize a segmentation system. The point was to hold the downstream probe still and watch what changes as the representation changes.

The evaluation asks four different questions:

1. **Retrieval:** if I choose a clean patch inside a region, do its nearest neighbors come from the same region?
2. **Clustering:** does the feature partition line up with the exact scene regions?
3. **Boundary quality:** do candidate boundaries fall near the known region boundaries?
4. **Grouping stability:** when the scene appearance changes, does the discrete partition stay similar?

Those are related questions, but they are not the same question.

That turned out to be the first important result.

## There is no single “best layer”

I expected depth to matter. I did not expect it to matter in such different ways for different measurements.

For DINOv2, same-region retrieval is strongest at block 5:

- block 2: `0.8000`;
- block 5: **`0.8375`**;
- block 11: `0.7250`.

But clustering alignment is strongest earlier:

- block 2 ARI: **`0.5323`**;
- block 5 ARI: `0.5011`;
- block 11 ARI: `0.4055`.

Boundary quality follows the same early-layer pattern, while grouping stability under the controlled perturbations is strongest at the final probed block: **`0.8887`** at block 11.

![Same-region retrieval for DINOv2 and SigLIP 2 across three transformer depths.](assets/figures/retrieval-by-depth.svg)

*Figure 2. Retrieval changes with depth, but the direction depends on the backbone. A deeper representation is not automatically a better spatial representation.*

SigLIP 2 tells a different story. On the base scene, its earliest probed block is strongest on all three direct spatial-alignment measures:

| Metric | Block 2 | Block 5 | Block 11 |
| --- | ---: | ---: | ---: |
| Same-region retrieval | **0.8125** | 0.7375 | 0.6250 |
| ARI | **0.6698** | 0.3701 | 0.1771 |
| Boundary F1 | **0.9291** | 0.7402 | 0.6074 |
| Grouping stability | **0.7854** | 0.6286 | 0.4882 |

The deeper features are not “worse” in some universal sense. They are optimized by training to become useful for the model's objectives. The point is narrower: **the depth that is useful for one spatial question may not be the depth that is useful for another.**

![Region alignment measured by Adjusted Rand Index across depth.](assets/figures/clustering-by-depth.svg)

*Figure 3. The fixed clustering probe exposes different depth profiles across the two backbones. The values are controlled-probe measurements, not a model leaderboard.*

This sounds obvious when written down. In practice, it is easy to forget. A foundation model often gets treated as one representation, even though a transformer is a sequence of representations.

If the application cares about boundaries, retrieval, robustness, or relationships, “which model?” is incomplete without “which layer, for which claim?”

## The towers exposed the gap between similarity and identity

The controlled scene contains two tower-like structures on purpose.

If representation similarity were already equivalent to object identity, a query on Tower A should reliably retrieve Tower B as the corresponding structure.

That is not what happened.

Some layers retrieved same-region patches extremely well while still failing to connect the two separate towers. In other words, the representation could be excellent at answering:

> “Which patches look like this patch in context?”

without answering:

> “Which other object is another instance of the same thing?”

That distinction is one of the most useful lessons from this experiment.

Similarity is continuous evidence. Identity is a semantic claim.

The first can support the second, but it does not define it.

## Then I tried to build an actual relationship graph

A region map is useful, but the goal of this article was not segmentation. I wanted a representation that could say something explicit about how candidate regions relate.

So after candidate regions were built, I added separate deterministic edge rules for:

- `adjacent` — components touch on the patch grid;
- `near` — components are within a fixed source-image distance without touching;
- `embedding_similar` — mean region features exceed a fixed cosine threshold.

Only after those edges existed did the exact scene masks enter for node matching and scoring.

This is where the experiment became more interesting.

The best selected public runs were weak:

| Model | Selected layer | Node recall | Adjacent F1 | Near F1 | Embedding-similar F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 | 11 | 0.80 | 0.340 | 0.167 | **0.000** | 0.169 |
| SigLIP 2 | 5 | 0.70 | 0.417 | 0.167 | **0.000** | 0.194 |

![Typed relationship F1 for the selected public runs.](assets/figures/typed-relationships.svg)

*Figure 4. Region recovery and relationship recovery are separate problems. The fixed embedding-similarity edge rule recovers none of the declared semantic-similarity edges in the selected runs.*

The failure is useful because it prevents a much more dangerous shortcut: collapsing everything into a generic `related` score.

If adjacency works sometimes, proximity works differently, and embedding similarity fails at the fixed threshold, averaging them into one relation number would make the graph look cleaner while making it harder to understand.

I would rather have an awkward graph whose edge types can fail independently than a polished graph whose semantics are impossible to audit.

## One relationship could not even be represented

The scene also contains a warehouse door that is geometrically contained by the warehouse.

The flat clustering representation cannot express that relationship at all.

Every patch belongs to one disjoint component. A component can sit beside another component, but it cannot simultaneously be a child region nested inside a parent region.

So I did not report a containment score.

I reported containment as **unsupported by the representation**.

That may be the most mundane result in the article, but I think it is an important engineering habit. When the data structure cannot express a claim, changing the threshold is not the solution. The representation itself has to change—probably toward hierarchical proposals or overlapping regions.

## A realistic scene made the same weakness easier to see

After the controlled evaluation, I ran the same fixed probe on a more realistic industrial-yard scene with repeated cabinets, depth variation, corrugated surfaces, fencing, vegetation, reflections, shadows, and partial occlusion.

There are no labels or masks for this scene, so I do not report accuracy.

I only inspect how the candidate partition behaves.

For DINOv2, the connected candidate-region count changes from **12 → 8 → 10** across blocks 2, 5, and 11.

For SigLIP 2, it changes from **7 → 7 → 26**.

![Candidate-region counts on the realistic qualitative stress test.](assets/figures/realistic-candidate-regions.svg)

*Figure 5. The same fixed probe changes character with representation depth on a cluttered scene. These counts are descriptive, not accuracy measurements.*

The exact numbers are not the conclusion. The important part is that depth changes the *kind* of partition the downstream system receives.

That matters if the next stage expects stable objects, persistent regions, or a graph whose nodes have consistent meaning. A pipeline can be deterministic after the transformer and still be unstable because its input representation changes character with depth.

## The public explorer is intentionally smaller than the research workspace

The [interactive evidence explorer](https://vlada22.github.io/grounded-relational-intelligence-public/) is a static public artifact.

It contains no model weights and performs no model inference. Instead, it lets you switch between DINOv2 and SigLIP 2, move across the three probed layers, compare retrieval, clustering, boundary, and stability measurements, inspect the typed-edge result, and click the controlled image to see the exact source patch coordinates.

That last detail matters more than it sounds.

A patch representation becomes much easier to reason about when every selected patch can be mapped back to an exact source box. The model may operate in embedding space, but the evidence should remain anchored to pixels.

The public repository is deliberately narrower than the private research environment. It contains the procedural scene, reviewed aggregate measurements, deterministic figure builders, the static explorer, and validation tests. It does **not** contain gated-model workflows, credentials, model weights, private feature archives, or licensed research materials.

That is not a loss of reproducibility. It is a clearer definition of what this public artifact promises to reproduce.

## What this experiment shows—and what it does not

For me, the strongest result is not that one transformer or one layer won.

It is that the path from features to relationships has several distinct failure points:

1. a patch representation may be locally useful but not instance-aware;
2. a clustering probe may recover some regions but miss others;
3. region recovery and edge recovery compound their errors;
4. different relationship types require different rules;
5. some relationships require a richer representation than a flat partition can provide.

The experiment also shows that feature depth is a design choice, not a harmless implementation detail.

What it does **not** show is equally important.

This is not a segmentation benchmark. It is not a model leaderboard. The controlled scene is small and synthetic. The realistic scene has no ground truth. The typed graph is intentionally simple. I did not train a proposal network, calibrate thresholds on a held-out dataset, or test long-range scene graphs across a large corpus.

Those would be reasonable next steps if the goal were state-of-the-art relationship detection.

That was not my goal here.

My goal was to find the boundary between what a frozen representation gives us and what an application still has to construct explicitly.

## The larger idea

Across the first three articles, the pattern is becoming clearer to me.

A model can observe a useful signal without owning the final claim.

In video, a segmentation model can observe an object, but a deterministic tool should still measure when it crosses a boundary.

In 3D reconstruction, a depth model can estimate geometry, but camera conventions, scale, visibility, and deterministic measurement still decide what distances can be trusted.

Here, a transformer can organize patches, but the application still has to decide what counts as a region, what kind of relationship an edge represents, and what evidence is strong enough to justify it.

That leads to a principle I expect to keep using:

> **Similarity is evidence. A relationship is a claim. Keep the two separate until the claim can be tested.**

The distinction may feel conservative, but it makes visual systems easier to debug, easier to explain, and much harder to fool with an attractive visualization.

The next step in this series is temporal: once regions and relationships change from frame to frame, how do we decide whether a relationship persisted, disappeared, or was never observable in the first place?

That is where static scene structure starts becoming memory.

I have published the safe public code, aggregate evidence, figures, and [interactive explorer](https://vlada22.github.io/grounded-relational-intelligence-public/) in the [Grounded Relational Intelligence public repository](https://github.com/vlada22/grounded-relational-intelligence-public).

Where would you draw the line between a useful visual similarity and a relationship you would trust a system to state as fact?

---

### Reproducibility note

This public repository intentionally reproduces the **public evidence layer**, not every private inference experiment. It contains a deterministic procedural scene, reviewed aggregate measurements from DINOv2 and SigLIP 2, scripts that regenerate the publication SVGs, a dependency-free static explorer, and tests that fail if gated-model or credential markers appear in the public tree. No model weights or raw full-dimensional feature archives are distributed.

### References

- Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), 2023.
- Tschannen et al., [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786), 2025.
