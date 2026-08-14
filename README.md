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

