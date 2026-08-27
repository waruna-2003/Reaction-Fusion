# Facebook Posts raw snapshot — 2026-08-14

## Provenance

- Original file: `C:\Users\USER\Downloads\facebook_posts (2).xlsx`
- Project copy: `data/raw/source_exports/facebook_posts_2026-08-14.xlsx`
- Imported on: 2026-08-14
- File size: 139,797 bytes
- SHA-256: `FC47C65055C06804DEFF9077C6CDC6FBCE971DA0CD14591876B83861E59D8032`
- Source workbook preserved unchanged: yes (source and project-copy hashes match)

The original collection source, collection method, date range, Facebook pages,
permissions/terms basis, and collector identity were not encoded in the workbook.
These must be added to the project data register before publication.

## Workbook structure

- Worksheet: `FB Posts`
- Data records: 1,000
- Columns: `#`, `Post Text`, `Likes`, `Love`, `Care`, `Haha`, `Wow`, `Sad`,
  `Angry`, and `Total Reactions`
- Formula cells: none

## Automated validation results

- All 1,000 rows contain text, an ID-like row number, every reaction count, and a total.
- All seven reaction fields contain valid, non-negative integer values.
- Every `Total Reactions` value equals the sum of the seven reaction counts.
- 984 records contain at least one Sinhala Unicode character.
- 16 records contain no Sinhala-script characters; these include Singlish and three
  placeholder-like records containing only `232`.
- 3 records have zero total reactions; all three contain `232` as text.
- 52 normalized text values occur more than once, producing 56 extra duplicate
  occurrences beyond the first copy.
- 44 duplicate-text groups have different reaction distributions. These may be
  repeated collection snapshots or distinct reposts, but the workbook lacks source
  identifiers and timestamps needed to distinguish those cases.
- 4 records contain a phone-number-like value. These must be masked or removed in
  processed/released datasets.
- No URLs or email-address patterns were detected by the automated scan.

## Reaction distribution observations

- Total reactions per row: median 305, mean 738.523, maximum 28,536.
- Dominant reaction by row: Like 527, Haha 425, Love 34, Sad 11, tie 3.
- Care, Wow, and Angry are present but are never the largest reaction in a row.
- Reaction counts are strongly skewed, so raw counts should not be used directly as
  sentiment scores. ReactionFusion should work with normalized proportions and
  explicitly handle low-frequency reactions and high-engagement outliers.

## Suitability decision

**Status: suitable as an initial raw ReactionFusion dataset snapshot, but not yet
suitable as a final model-training or benchmark dataset.**

The workbook directly supports the core ReactionFusion idea because it contains
Sinhala social-media text and all seven Facebook reaction counts. Its reaction
fields are unusually complete and internally consistent. It is appropriate for
exploratory reaction-distribution analysis, development of validation rules, and
early versions of the fusion algorithm.

Before training or reporting research results, a processed version must:

1. Resolve placeholders and zero-reaction records.
2. Distinguish true reposts from duplicated collection snapshots.
3. Mask phone numbers and review text for other personal identifiers.
4. Assign explicit language labels (`sinhala`, `singlish`, `mixed`, `other`).
5. Add stable source/post grouping identifiers and timestamps where ethically and
   legally permissible, so train/test leakage can be prevented.
6. Record source page/category and collection provenance to measure source bias.
7. Generate versioned ReactionFusion labels and confidence scores only after the
   labeling rules are frozen.
8. Create a separately human-annotated evaluation subset.

This snapshot contains posts, not a demonstrated comment dataset. A classifier
trained only on these records may not generalize to the planned comment-analysis
plugin. Comment-level Sinhala data should therefore be collected and evaluated as
a separate domain, or the platform's claims should be limited to post sentiment.

