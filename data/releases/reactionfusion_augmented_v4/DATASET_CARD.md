# ReactionFusion combined augmentation v4 dataset card

## Status

This release combines the legacy 997-record source with 15,000 explicitly
synthetic posts. Synthetic and legacy records remain identifiable through
`data_origin`, `text_provenance`, and `reaction_provenance`.

The synthetic workbook states that its Facebook URLs, reaction counts, and post
text are fabricated. Its supplied annotations are stored as unverified synthetic
supervision and are **not human ground truth**.

## Counts

- Combined records: 15997
- Legacy records: 997
- Synthetic records: 15000
- Deduplicated records: 15943
- Original human development annotations: 120
- Provided synthetic annotations: 15,000

## Restarted predictions

ReactionFusion v1, v2, and neural v3 predictions were regenerated for the combined
records. Existing grouped out-of-fold predictions were retained for the original
120 human-development records.

## Synthetic transfer experiment

A neural ensemble trained on the 15,000 synthetic annotations achieved
0.137 accuracy and 0.117
four-class macro-F1 on the original adjudicated human development records. The
experiment is rejected for model promotion because it performs substantially below
the existing human-calibrated neural v3 model.

## Usage policy

- Use `legacy_source` records for the original research lineage.
- Use `synthetic_augmentation` only for explicitly reported augmentation studies.
- Never report synthetic URLs or reactions as observed Facebook engagement.
- Do not combine synthetic annotations with a final human benchmark.
