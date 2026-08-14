# Dataset protocol

## Required record fields

Keep stable identifiers and a documented schema. A useful initial record contains:

| Field | Purpose |
|---|---|
| `record_id` | Internal, non-identifying stable ID |
| `text` | Sinhala post/comment text after permitted collection |
| `like_count` ... `care_count` | Counts for all seven reactions |
| `created_at` | Optional timestamp, generalized when necessary |
| `source_split_group` | Post/page grouping key used to prevent leakage |
| `label` | Generated sentiment label |
| `label_confidence` | ReactionFusion confidence score |
| `label_version` | Exact labeling algorithm/configuration version |

Store original platform IDs separately only when they are legally and ethically
required, and never expose personal identifiers in committed samples.

## Rules

1. Treat `data/raw/` as immutable.
2. Record collection date, source, consent/terms basis, and preprocessing version.
3. Remove names, profile links, user IDs, and other personal identifiers.
4. Deduplicate before splitting.
5. Split by post/source group—not random comment alone—to reduce data leakage.
6. Freeze the held-out human-annotated test set before tuning ReactionFusion.
7. Version every label with algorithm parameters and dataset revision.
8. Keep a manually annotated validation subset with inter-annotator agreement.

## Implemented release files

```text
data/releases/reactionfusion_v1/
  dataset.csv
  dataset_deduplicated.csv
  train.csv
  validation.csv
  test.csv
  human_annotation_candidates.csv
  human_annotation_workbook.xlsx
  rejected_records.csv
  quality_report.json
  labeling_config.json
  DATASET_CARD.md
```

`dataset.csv` retains repeated records for the main experiment, while
`dataset_deduplicated.csv` supports a deduplication ablation. All occurrences of
the same normalized text receive the same split, so duplicates cannot leak across
training, validation, and test sets.

The annotation workbook is deliberately blinded. Annotators see masked post text
and language type, but no reaction counts, automatic label, score, or algorithmic
confidence. Two annotators independently assign one sentiment label, multi-label
emotions, approval stance, sarcasm, and annotation confidence. The sentiment
choices are `positive`, `negative`, `neutral`, `mixed`, or `uncertain`; emotion,
approval, and sarcasm fields use `yes`, `no`, or `uncertain`. An adjudicator then
resolves disagreements in a separate sheet.

Do not publish raw social-media data until licensing, platform terms, privacy,
and institutional ethics requirements have been reviewed.
