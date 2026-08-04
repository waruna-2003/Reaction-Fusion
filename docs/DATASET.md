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

## Suggested processed files

```text
data/processed/
  reactionfusion_v1/
    train.parquet
    validation.parquet
    test.parquet
    schema.json
    dataset_card.md
```

Do not publish raw social-media data until licensing, platform terms, privacy,
and institutional ethics requirements have been reviewed.

