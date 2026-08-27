# ReactionFusion

ReactionFusion is a Sinhala social-media sentiment-analysis research project that
uses the complete Facebook reaction distribution—including Like, Love, Care,
Haha, Wow, Sad, and Angry—to generate sentiment labels without discarding
ambiguous reactions.

## Current state

The project has restarted with a new dataset of 2,904 original Facebook posts.
Human annotation is currently in progress using one additional `sentiment` column
with the labels `positive`, `negative`, `neutral`, `mixed`, and `uncertain`.

- Source: `data/raw/original_exports/facebook_posts_original_2904.xlsx`
- Annotation workbook: `data/annotations/original_posts_v1/annotator_01.xlsx`
- Records with blank post text: 0
- Repeated normalized post-text rows: 277

The repeated rows are retained in the immutable source. They must be grouped by
normalized post text before train/validation/test splitting to prevent leakage.

## Model evolution

The earlier experimental models remain in the repository as research evidence:

| Version | Main idea | Development accuracy | Development macro F1 |
|---|---|---:|---:|
| V1 | Hand-designed fusion rules using every reaction | 0.368 | 0.290 |
| V2 | Human-calibrated hybrid statistical fusion | 0.479 | 0.429 |
| V3 | Reaction-only multi-task neural ensemble | 0.496 | 0.436 |
| V4 | Synthetic augmentation and transfer experiment | 0.237* | 0.154* |
| V5 | Provenance-weighted human and synthetic neural model | 0.538 | 0.422 |

`*` V4 values are evaluation against unverified synthetic labels and are not a
human benchmark. The historical results are development measurements, mainly on
an uncertainty-enriched set of 120 human records, and must not be presented as
final generalization performance.

See `experiments/model_evolution/README.md` for the experiment narrative,
limitations, model snapshots, and supporting reports.

## Active research workflow

1. Complete human sentiment annotation of the new original-post dataset.
2. Audit missing labels, invalid categories, duplicates, and class balance.
3. Freeze leakage-safe training, validation, and representative test groups.
4. Train a new reaction-only sentiment model using real human labels.
5. Compare it fairly with V1–V5 and reaction-filtering baselines.
6. Train and compare BiLSTM, GRU, mBERT, and XLM-R on fixed dataset splits.
7. Integrate the selected model into the demonstration platform.

## Repository layout

```text
configs/                 Data and model configurations
data/                    Current source, annotation, and future processed data
docs/                    Dataset protocol and research documentation
experiments/             Historical model-evolution evidence
models/experimental/     Serialized historical ReactionFusion models
platform/                Demonstration platform scaffolding
reports/                 Future figures and thesis-ready tables
scripts/                 Reproducible command-line entry points
src/reactionfusion/      Research implementation, including V1–V5 experiments
tests/                   Automated checks for the research implementation
```

Do not use the earlier experimental models as the final benchmark. Their purpose
is to document how ReactionFusion evolved and motivate the new real-data study.
