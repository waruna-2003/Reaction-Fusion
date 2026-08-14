# Dataset preprocessing

## Run

From the repository root:

```powershell
python scripts/preprocess_dataset.py
```

The raw workbook is immutable and ignored by Git. The script reads
`data/raw/source_exports/facebook_posts_2026-08-14.xlsx` and writes private
intermediate files plus the privacy-reviewed release in
`data/releases/reactionfusion_v1/`.

## Pipeline

1. Validate the required workbook sheet and columns.
2. Coerce all seven reaction counts to non-negative integers and reconcile totals.
3. Reject missing/placeholder text and zero-engagement records with reason codes.
4. Normalize Unicode to NFC, remove invisible characters, and collapse whitespace.
5. Preserve Sinhala characters, emoji, punctuation, and code-mixed text.
6. Mask phone numbers, email addresses, URLs, and user handles.
7. Assign heuristic language categories: Sinhala, mixed, Singlish, or other.
8. Identify exact normalized duplicates with a stable text hash.
9. Create raw and Laplace-smoothed reaction ratios, dominance, entropy, margin,
   engagement, ReactionFusion v1 provisional labels, and filtered-baseline labels.
10. Create deterministic 70/15/15 group-aware splits. A normalized text hash is
    never assigned to more than one split.
11. Select high-uncertainty examples for blinded independent human annotation.

## ReactionFusion v1 provisional method

Love and Care provide positive anchor evidence; Sad and Angry provide negative
anchor evidence. Like, Haha, and Wow are retained and receive context-dependent
polarity derived from the clear-reaction balance. The distribution score is
combined with entropy, opposing evidence, and engagement to assign a provisional
label and confidence. Exact parameters are versioned in
`configs/labeling/reactionfusion_v1.json` and copied into every release.

This is a research hypothesis pending human validation. Do not present the
generated label as gold-standard sentiment, and do not tune the method against the
held-out human-adjudicated set.

## Collaboration

Upload `human_annotation_workbook.xlsx` to Google Drive and open it with Google
Sheets. The coordinator should give each reviewer a separate copy or restrict the
other reviewer sheet until both independent passes are complete. Each reviewer
records sentiment, multi-label emotions, approval, sarcasm, and confidence. Merge
their decisions into the Adjudication sheet, download the completed workbook, and
preserve it as a versioned validation artifact.
