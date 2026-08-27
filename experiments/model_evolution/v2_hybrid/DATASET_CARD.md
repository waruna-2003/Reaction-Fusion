# ReactionFusion v2 development dataset card

## Status

This is an **experimental development release**, not a final benchmark dataset.
ReactionFusion v2 is a reaction-only hybrid model calibrated with 120
uncertainty-enriched human annotations. It does not use post text as an input.

## Architecture

- Laplace-smoothed ratios for all seven reactions.
- Clear-valence anchors and context gates for Like, Haha, and Wow.
- Entropy, opposition, dominance, engagement, and interaction features.
- Eleven regularized human-calibrated emotion/stance probability models.
- Four-class regularized sentiment fusion: negative, neutral, positive, mixed.
- Temperature calibration and confidence-based abstention.
- Cross-fitted labels for all human-calibration records to prevent in-sample labels.

## Cross-validated development results

- Evaluation: grouped out-of-fold predictions on the annotated difficult-case set.
- Abstaining-label accuracy: 0.419
- Abstaining-label macro F1: 0.391
- Observed confident coverage: 0.846
- Accuracy on covered records: 0.495
- Group leakage: 0

| Method | Accuracy | Four-class macro F1 |
|---|---:|---:|
| ReactionFusion v2 candidate | 0.479 | 0.429 |
| ReactionFusion v1 | 0.368 | 0.290 |
| Filtered baseline | 0.427 | 0.278 |

Paired exact McNemar p-values are 0.111 for v2 versus v1 and
0.471 for v2 versus the filtered baseline. These development-set
statistics do not replace evaluation on a representative frozen human test set.

## Generated label distribution

{"uncertain": 107, "negative": 485, "neutral": 96, "positive": 237, "mixed": 72}

## Limitations

- The calibration sample was selected for uncertainty and is not representative.
- Only nine adjudicated mixed examples are available.
- Surprise, fear, and care/empathy have very few positive human examples; their
  probabilities are exploratory rather than validated emotion classifiers.
- V2 must be evaluated once on a separately frozen representative human test set.
- Records labeled `uncertain` should be excluded from downstream ANN training.
- Source licensing, privacy, platform terms, and ethics requirements still apply.

## Files

- `dataset.csv`: complete v2 development release with v1 audit columns.
- `dataset_deduplicated.csv`: one record per normalized text.
- `train.csv`, `validation.csv`, `test.csv`: existing leakage-safe splits.
- `model.json`: serialized v2 feature, emotion, sentiment, and calibration model.
- `training_config.json`: frozen training hyperparameters.
- `cross_validation_results.json`: complete grouped out-of-fold metrics.
- `human_validation_oof.csv`: record-level out-of-fold predictions.
- `quality_report.json`: generated release counts and distributions.
