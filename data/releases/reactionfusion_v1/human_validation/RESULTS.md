# ReactionFusion v1 human-validation results

## Scope

The completed annotation workbook contains 120 uncertainty-enriched
records selected by the ReactionFusion v1 annotation-candidate procedure. These
results describe performance on difficult/ambiguous cases and must **not** be
reported as an unbiased estimate for the full dataset.

Three adjudicated `uncertain` records are excluded from classifier metrics, leaving
117 evaluated records.

## Annotation quality

- Sentiment raw agreement: 0.725
- Sentiment Cohen's kappa: 0.594
- Sentiment disagreements: 33
- Final sentiment distribution: negative=48,
  neutral=35, positive=25,
  mixed=9, uncertain=3

## Provisional comparison

| Method | Accuracy | Macro F1 (4 classes) |
|---|---:|---:|
| ReactionFusion v1 | 0.368 | 0.290 |
| Filtered-reaction baseline | 0.427 | 0.278 |

ReactionFusion v1 has higher macro F1 on this hard subset, while the filtered
baseline has higher accuracy. Neither method predicts the human `mixed` class in
the current configuration, so mixed-class recall is zero. The result does not yet
support a general claim that ReactionFusion is superior. The paired exact McNemar
test gives p=0.210, so the accuracy difference is
not statistically significant at the 0.05 level. Use these errors to design v2 and
evaluate once on a separately frozen representative human test set.

## Next research action

Use the adjudicated development annotations to analyze failure patterns and tune a
versioned ReactionFusion v2 configuration. Do not overwrite v1. Freeze a separate,
representative human test sample before final model or algorithm comparison.
