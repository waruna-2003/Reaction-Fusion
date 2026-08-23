# ReactionFusion neural v3 architecture

## Purpose

Neural v3 tests whether nonlinear interaction learning improves automatic labels
generated from Facebook reaction distributions. It remains reaction-only: Sinhala
post text is never an input to the label generator.

## Network

```text
7 reaction counts
       |
27 smoothed distribution and interaction features
       |
standardization + small feature noise during training
       |
five independently initialized 16-unit tanh networks
       |
       +---- four-class sentiment softmax
       |
       +---- eleven emotion/stance sigmoid outputs
       |
probability averaging + temperature calibration + abstention
```

The multi-task emotion head supplies additional supervision to the shared hidden
layer. Class weighting, label smoothing, L2 regularization, feature noise, and a
five-member ensemble limit overfitting on the small annotation set.

## Evaluation

The pipeline uses repeated five-fold cross-validation grouped by normalized text
hash. Every annotated record in the released dataset receives averaged out-of-fold
predictions rather than an in-sample prediction.

Run:

```powershell
python scripts/train_reactionfusion_neural.py
```

The generated artifacts are written to
`data/releases/reactionfusion_neural_v3/`.

## Interpretation

The primary fold configuration reaches 0.556 accuracy and 0.485 four-class
macro-F1, compared with 0.479 and 0.429 for v2. Across three fold assignments,
however, mean accuracy is 0.504 and mean macro-F1 is 0.452. The averaged repeated
out-of-fold prediction reaches 0.496 accuracy and 0.436 macro-F1.

This improvement is small and fold-sensitive. It demonstrates a functioning
neural architecture, not proof of superiority. The next research priority is more
representative human annotation, particularly for mixed sentiment and rare
emotion targets, followed by a frozen external human test set.
