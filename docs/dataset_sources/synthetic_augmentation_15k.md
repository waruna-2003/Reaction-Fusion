# Synthetic 15k augmentation source

## Files

- Local raw posts: `data/raw/synthetic_exports/facebook_posts_synthetic_15k.xlsx`
- Provided annotations: `data/annotations/synthetic_15k/adjudicated_synthetic_15k.xlsx`
- Combined release: `data/releases/reactionfusion_augmented_v4/`

## Provenance

The posts workbook README states that all Facebook URLs and reaction counts are
fabricated and that the Sinhala/Singlish captions are template-generated synthetic
text. The combined release therefore marks every added row with:

- `data_origin = synthetic_augmentation`
- `text_provenance = template_generated_synthetic`
- `reaction_provenance = fabricated_synthetic`
- `annotation_provenance = provided_synthetic_unverified`

The raw workbook remains excluded by the repository raw-data policy; its SHA-256
is recorded in `source_audit.json` so a local copy can be verified.

## Validation

- 15,000 post rows and 15,000 rows in each annotation sheet.
- Unique and aligned `SYN_POST_######` identifiers.
- Exact normalized-text alignment across posts and all annotation sheets.
- No reaction-total mismatches.
- No duplicate synthetic source row numbers.
- Zero normalized-text group leakage across combined train/validation/test splits.

## Model decision

The 15,000 synthetic labels were tested as neural supervision against the original
120-record adjudicated human development set. Accuracy was 0.137 and four-class
macro-F1 was 0.117. The synthetic-trained model was rejected for promotion, while
the records remain available for explicitly reported augmentation studies.
