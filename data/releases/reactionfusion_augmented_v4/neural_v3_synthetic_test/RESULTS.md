# Neural v3 evaluation on the synthetic augmentation set

## Evaluation design

The frozen human-calibrated neural v3 model was evaluated without retraining on
15,000 synthetic records. Reference labels are the supplied synthetic adjudication
labels and are not verified human ground truth.

| Method | Accuracy | Four-class macro F1 |
|---|---:|---:|
| Neural v3 candidate | 0.237 | 0.154 |
| Neural v3 with abstention | 0.230 | 0.149 |
| ReactionFusion v2 candidate | 0.269 | 0.203 |
| ReactionFusion v1 | 0.221 | 0.206 |
| Filtered baseline | 0.192 | 0.158 |

## Neural v3 diagnostics

- Confident coverage: 0.957
- Accuracy among confident predictions: 0.240
- Mean candidate confidence: 0.634
- Multiclass log loss: 1.993
- Multiclass Brier score: 1.001
- Expected calibration error: 0.398

## Interpretation

Neural v3 predicts neutral for most synthetic records and performs below v2 on the
provided synthetic labels. This is evidence of domain shift between the original
human-calibration sample and the fabricated reaction/text generation process. It
does not invalidate the original human evaluation, and it does not validate the
synthetic annotations as research ground truth.
