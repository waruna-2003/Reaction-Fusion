# Synthetic 15k annotation workbook

`adjudicated_synthetic_15k.xlsx` contains 15,000 provided annotations aligned with
the synthetic post workbook in `data/raw/synthetic_exports/`.

These labels are retained as **provided synthetic annotations, unverified**. They
must not be described as human ground truth because:

- the associated post text, URLs, and reaction counts are explicitly synthetic;
- raw Annotator 1/Annotator 2 sentiment agreement is 48.36% (kappa 0.348);
- when the two annotators disagree, adjudication follows Annotator 2 in 97.34% of
  cases; and
- annotation notes are heavily repeated/template-like.

Use these labels only in clearly identified augmentation experiments. The original
120-record adjudicated workbook remains the human development reference.
