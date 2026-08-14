# ReactionFusion v1 provisional dataset card

## Status

This is a privacy-masked research release generated from 1,000 collected Facebook
post records. ReactionFusion labels are **provisional weak labels** and must not be
treated as human ground truth. Complete the included annotation task before final
algorithm validation or benchmark reporting.

## Processing summary

- Source records: 1000
- Accepted records: 997
- Rejected records: 3
- Records with privacy masking: 4
- Duplicate groups: 51
- Extra duplicate occurrences: 54
- Deduplicated records: 943
- Human annotation candidates: 120
- Label version: `reactionfusion_v1_provisional`

## Intended use

Development and validation of Sinhala social-media sentiment labeling and model
experiments. Identical normalized texts are assigned to the same split to prevent
duplicate leakage. Use `dataset_deduplicated.csv` for the deduplication ablation.

## Important limitations

- Source post/page identifiers, timestamps, and collection provenance are absent.
- Duplicate texts may represent reposts or repeated collection snapshots.
- The dataset contains posts; performance on comments must be evaluated separately.
- Language categories are heuristic.
- The reaction-fusion weights are hypotheses pending human validation.
- Source licensing, platform terms, and institutional ethics approval must be
  confirmed before redistribution beyond this research repository.

## Files

- `dataset.csv`: accepted masked records with weak labels and split assignments.
- `dataset_deduplicated.csv`: highest-engagement record per normalized text.
- `train.csv`, `validation.csv`, `test.csv`: group-aware splits.
- `human_annotation_candidates.csv`: blank two-annotator/adjudication task.
- `human_annotation_workbook.xlsx`: Google Sheets-ready blinded annotation workbook.
- `rejected_records.csv`: non-sensitive rejection audit.
- `quality_report.json`: reproducible counts and distributions.
- `labeling_config.json`: exact provisional ReactionFusion parameters.
