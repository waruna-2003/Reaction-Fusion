# ReactionFusion model evolution

This directory preserves the evidence needed to explain how ReactionFusion evolved.
The associated legacy datasets have been removed from the active repository state.

## V1 — reaction-fusion rules

V1 converted all seven reaction proportions into provisional sentiment using
hand-designed valence, ambiguity, entropy, opposition, and confidence rules. On
117 adjudicated difficult cases, it reached 0.368 accuracy and 0.290 macro F1.
It improved macro F1 over the filtered-reaction baseline but failed to recognize
the mixed class reliably.

## V2 — human-calibrated hybrid model

V2 replaced fixed fusion weights with regularized statistical calibration learned
from 120 human-annotated difficult cases. It used 27 reaction-distribution features,
temperature calibration, and abstention. Its candidate predictions reached 0.479
accuracy and 0.429 macro F1.

## V3 — multi-task neural ensemble

V3 introduced a five-member neural ensemble with a 16-unit hidden layer and
joint sentiment/emotion learning. Averaged repeated out-of-fold performance was
0.496 accuracy and 0.436 macro F1. The gain over V2 was small and not statistically
significant, showing that additional real annotations were needed.

## V4 — synthetic augmentation experiment

V4 tested 15,000 fabricated, template-generated posts and supplied synthetic
labels. The frozen V3 model achieved 0.237 accuracy and 0.154 macro F1 against
those synthetic labels. A synthetic-trained transfer probe performed poorly on
real human labels, demonstrating substantial domain shift. Synthetic data was
therefore rejected as the primary training source.

## V5 — provenance-weighted combined neural model

V5 jointly trained on human and strongly down-weighted synthetic annotations. Its
averaged human out-of-fold result reached 0.538 accuracy but only 0.422 macro F1,
and mixed-class recall remained zero. V5 improved accuracy but did not improve the
balanced metric over V3, so it was not accepted as a superior final model.

## Presentation conclusion

The sequence shows a genuine research progression: complete-reaction rules,
human calibration, nonlinear neural fusion, synthetic-data stress testing, and
provenance-weighted learning. It also establishes the reason for the current
restart: the next model must be trained and evaluated on a larger representative
set of original posts with real human sentiment labels.

Supporting JSON reports and dataset cards are stored in each version directory.
Serialized snapshots are stored in `models/experimental/`.
