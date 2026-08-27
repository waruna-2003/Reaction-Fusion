# ReactionFusion neural v3 development dataset card

## Status

This is an **experimental development release**, not a final benchmark. The neural
architecture and its hyperparameters were developed using the same 120-record
uncertainty-enriched annotation set used for evaluation.

## Model

- Inputs: reaction counts and 27 derived distribution features; no post text.
- Five-member ensemble with one 16-unit tanh hidden layer per member.
- Multi-task outputs: four-class sentiment and eleven emotion/stance probabilities.
- Class weighting, label smoothing, L2 regularization, feature noise, temperature
  calibration, and confidence-based abstention.
- Repeated grouped out-of-fold labels for every human-calibration record.

## Development results

| Evaluation | Accuracy | Four-class macro F1 |
|---|---:|---:|
| Neural primary grouped five-fold | 0.556 | 0.485 |
| Neural averaged repeated OOF | 0.496 | 0.436 |
| ReactionFusion v2 | 0.479 | 0.429 |

Across fold seeds, mean accuracy is 0.504 and mean
macro F1 is 0.452. The exact paired McNemar p-value
for averaged neural predictions versus v2 is 0.839.

## Generated release

- Records: 997
- Deduplicated records: 943
- Labels: {"uncertain": 113, "negative": 522, "positive": 252, "neutral": 94, "mixed": 16}

## Research limitation

The neural model provides only a small and fold-sensitive development improvement.
More representative human annotations—especially mixed, neutral, and rare emotion
examples—are required before claiming that the neural architecture outperforms v2.
Do not use the existing 120 records as the final research test set.
