# ReactionFusion

ReactionFusion is a Sinhala social-media sentiment-analysis research project that
uses the complete Facebook reaction distribution to generate automatic sentiment
labels. The repository also contains model experiments and a Facebook-like
demonstration platform with a sentiment analytics plugin.

## Repository layout

```text
configs/                 Experiment and labelling configurations
data/                    Raw inputs and versioned, privacy-reviewed dataset releases
docs/                    Research protocol, data dictionary, and documentation
experiments/             Versioned experiment definitions and result summaries
models/                  Local trained model files
notebooks/               Numbered exploratory notebooks only
reports/                 Figures, tables, and thesis-ready outputs
scripts/                 Reproducible command-line pipeline entry points
src/reactionfusion/      Core Python research package
tests/                   Automated tests
platform/backend/        Demonstration platform API
platform/frontend/       Demonstration platform UI
```

## Dataset lifecycle

```text
data/raw -> data/interim -> data/processed
```

- `raw`: immutable source export; never edit it manually.
- `interim`: cleaned, normalized, and deduplicated records.
- `processed`: model-ready splits and ReactionFusion-generated labels.
- `external`: separately sourced resources such as Sinhala stop-word lists.
- `samples`: tiny, anonymized examples safe to commit for tests/documentation.

GitHub tracks source code, configurations, documentation, compact experiment
results, and dataset releases that have passed privacy and licensing review. Raw
social-media exports remain local and ignored by Git. Large future datasets and
model artefacts require a separate storage decision before publication.

## Recommended workflow

1. Place a source export in `data/raw/` and record its provenance.
2. Validate and anonymize it using scripts in `scripts/`.
3. Clean text and generate ReactionFusion labels using versioned configuration.
4. Create fixed train/validation/test splits in `data/processed/`.
5. Train and evaluate each model using the same split and evaluation protocol.
6. Store metrics in `experiments/` and generated charts in `reports/figures/`.

See [docs/DATASET.md](docs/DATASET.md) before adding data.

## Current dataset preprocessing

The initial source snapshot contains 1,000 Facebook posts with counts for Like,
Love, Care, Haha, Wow, Sad, and Angry. Run the complete deterministic pipeline
from the repository root:

```powershell
python scripts/preprocess_dataset.py
```

The pipeline validates the schema and reaction totals, rejects invalid placeholder
records, normalizes Sinhala text while preserving emoji, masks common personal
identifiers, detects duplicates, creates reaction-distribution features, assigns
provisional ReactionFusion and filtered-baseline labels, and makes group-aware
70/15/15 splits. Public, privacy-masked outputs are written to
`data/releases/reactionfusion_v1/`.

The labels are weak labels, not human ground truth. Use
`human_annotation_workbook.xlsx` for independent double annotation and
adjudication before reporting the algorithm as validated.

After both reviewers and adjudication are complete, run:

```powershell
python scripts/evaluate_human_annotations.py
```

This validates the completed workbook, calculates inter-annotator agreement, and
compares ReactionFusion v1 and the filtered-reaction baseline against adjudicated
human sentiment labels.

## ReactionFusion v2 development model

Train the reaction-only, human-calibrated hybrid v2 model with:

```powershell
python scripts/train_reactionfusion_v2.py
```

The command performs grouped out-of-fold evaluation, trains emotion and sentiment
calibrators, applies confidence-based abstention, serializes the model, and creates
`data/releases/reactionfusion_v2/`. See
[docs/REACTIONFUSION_V2.md](docs/REACTIONFUSION_V2.md) for its architecture and
research limitations.

## ReactionFusion neural v3

Train the reaction-only multi-task neural ensemble with:

```powershell
python scripts/train_reactionfusion_neural.py
```

The pipeline runs repeated grouped cross-validation, serializes the neural
ensemble, and creates `data/releases/reactionfusion_neural_v3/`. See
[docs/REACTIONFUSION_NEURAL_V3.md](docs/REACTIONFUSION_NEURAL_V3.md) for the
architecture, development results, and limitations.

## Synthetic augmentation restart

The guarded v4 merge combines the 997 legacy records with 15,000 explicitly
synthetic posts while preserving source and annotation provenance:

```powershell
python scripts/merge_augmented_dataset.py
```

Outputs are written to `data/releases/reactionfusion_augmented_v4/`. The supplied
synthetic annotations are retained for augmentation experiments but are not treated
as human ground truth. See
[docs/dataset_sources/synthetic_augmentation_15k.md](docs/dataset_sources/synthetic_augmentation_15k.md).

Test the frozen neural v3 model on all synthetic records without retraining:

```powershell
python scripts/evaluate_neural_v3_synthetic.py
```

Results are written to
`data/releases/reactionfusion_augmented_v4/neural_v3_synthetic_test/`.

Train the provenance-weighted combined neural model using all synthetic and legacy
annotations:

```powershell
python scripts/train_reactionfusion_combined.py
```

The combined v5 release is written to
`data/releases/reactionfusion_neural_combined_v5/`.

