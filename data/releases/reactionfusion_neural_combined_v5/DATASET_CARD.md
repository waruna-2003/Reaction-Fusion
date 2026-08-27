# ReactionFusion neural combined v5 dataset card

## Training design

All 15,000 supplied synthetic annotations and all 120 legacy human annotations
participate in joint multi-task neural training. Provenance weights are 0.0001 for
each synthetic record and 1.0 for each human record because the earlier synthetic
transfer experiment failed on human labels.

Human evaluation remains leakage-safe: each human record is predicted out-of-fold,
while all synthetic records are training-only augmentation. There is zero
normalized-text overlap between the two origins.

## Human development results

| Model/evaluation | Accuracy | Four-class macro F1 |
|---|---:|---:|
| Combined v5 primary five-fold | 0.573 | 0.439 |
| Combined v5 averaged repeated OOF | 0.538 | 0.422 |
| Human-only neural v3 | 0.496 | 0.436 |

The exact paired McNemar p-value for combined v5 versus neural v3 is
0.383. These are development statistics
on the original uncertainty-enriched human sample, not final benchmark results.

## Release

- Records: 15997
- Human OOF records: 120
- Synthetic records used in training: 15000
- Uncertain output records: 575

Synthetic labels remain unverified and must not be described as human ground truth.
