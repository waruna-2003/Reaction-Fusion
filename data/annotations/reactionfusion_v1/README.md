# ReactionFusion v1 completed human annotations

## Source files

- `adjudication_completed.xlsx`: authoritative workbook containing the completed
  Annotator 1, Annotator 2, and final Adjudication sheets for 120 records.
- `annotator_02_completed.xlsx`: separately submitted reviewer workbook retained
  for provenance. Its Annotator 1 and Annotator 2 sheets are identical to the
  corresponding sheets in the authoritative workbook; its Adjudication sheet is
  blank.

The original files were supplied on 2026-08-23 and renamed only for consistent
repository organization. Do not edit these source workbooks. Generate derived
labels and metrics with:

```powershell
python scripts/evaluate_human_annotations.py
```

The 120 examples were selected as uncertain ReactionFusion v1 cases. They are an
uncertainty-enriched development/validation subset, not a random representative
sample of the complete dataset.
