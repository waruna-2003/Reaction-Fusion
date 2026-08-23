# ReactionFusion v2 hybrid architecture

## Status

ReactionFusion v2 is a development model calibrated with the completed 120-record
human annotation set. That set intentionally concentrates on uncertain v1 cases,
so v2 is not yet a final benchmark model.

## Non-circular design

V2 accepts only Facebook reaction counts and derived reaction-distribution
features. It never uses Sinhala post text. Human text annotations supervise model
calibration, while the resulting reaction-generated labels can later train ANN,
BiLSTM, GRU, mBERT, and XLM-R text classifiers without feeding their own text back
into the labeling algorithm.

## Architecture

1. Validate Like, Love, Care, Haha, Wow, Sad, and Angry counts.
2. Apply Laplace smoothing and calculate all seven reaction proportions.
3. Derive entropy, dominance, engagement, positive/negative anchors, opposition,
   mixed evidence, and contextual Like/Haha/Wow interaction features.
4. Estimate eleven human-calibrated probabilities: joy, affection, amusement,
   surprise, sadness, anger, care/empathy, fear, disgust, approval, and sarcasm.
5. Fuse reaction and emotion evidence through a regularized four-class softmax
   model for negative, neutral, positive, and mixed sentiment.
6. Apply out-of-fold temperature calibration.
7. Mark low-confidence records as `uncertain`; preserve the top four-class
   candidate label for auditing.

## Training and release

Run from the repository root:

```powershell
python scripts/train_reactionfusion_v2.py
```

The pipeline performs deterministic five-fold cross-validation grouped by
normalized text hash, selects calibration and abstention parameters from
out-of-fold predictions, trains the final model on all development annotations,
and writes `data/releases/reactionfusion_v2/`.

## Downstream text-model policy

Use only confident v2 records for ANN training:

```python
training = dataset[dataset["sentiment_label"] != "uncertain"]
```

Keep the existing train/validation/test group assignments. Do not randomly split
the combined dataset again. The v2 human-annotation set is for development and
error analysis; collect and freeze a separate representative human test sample
before making final research claims.
