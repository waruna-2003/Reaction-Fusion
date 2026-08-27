# Dataset protocol

## Current source

The active source is `data/raw/original_exports/facebook_posts_original_2904.xlsx`.
It contains 2,904 Facebook posts and the following columns:

| Field | Purpose |
|---|---|
| `#` | Source row identifier |
| `Post Text` | Original Sinhala or mixed-language post text |
| `Likes` | Like reaction count |
| `Love` | Love reaction count |
| `Care` | Care reaction count |
| `Haha` | Haha reaction count |
| `Wow` | Wow reaction count |
| `Sad` | Sad reaction count |
| `Angry` | Angry reaction count |
| `Total Reactions` | Sum of the seven reaction counts |

The immutable source workbook must not be edited during annotation.

## Human annotation

The working file is `data/annotations/original_posts_v1/annotator_01.xlsx`.
It preserves all source columns and adds only `sentiment`.

Allowed labels:

- `positive`: mainly favourable or happy meaning
- `negative`: mainly unfavourable, angry, harmful, or sad meaning
- `neutral`: primarily factual without a clear positive or negative position
- `mixed`: meaningful positive and negative sentiment occur together
- `uncertain`: the annotator cannot assign one of the four research classes reliably

Do not infer sentiment from reaction counts alone while manually annotating. Read
the post text and apply the same label definitions consistently.

## Quality and splitting rules

1. Preserve the raw source unchanged.
2. Validate that every annotation is one of the five allowed values.
3. Review missing labels and contradictory labels before training.
4. Group repeated normalized post text before splitting; 277 repeated-text rows
   were detected in the source audit.
5. Freeze a representative human-labeled test set before tuning the next model.
6. Never place duplicates of the same normalized post text in different splits.
7. Report class support, macro F1, per-class precision/recall, accuracy, confusion
   matrices, and calibration—not accuracy alone.
8. A single annotator is acceptable for initial development. Before final research
   claims, obtain a second independent annotation for a representative subset and
   report inter-annotator agreement.

Earlier synthetic and legacy datasets are not part of the active dataset. Their
model reports are retained only under `experiments/model_evolution/`.
