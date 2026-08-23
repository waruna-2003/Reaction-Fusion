"""Import completed annotations and evaluate ReactionFusion against humans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SHEETS = ("Annotator 1", "Annotator 2", "Adjudication")
IDENTITY_COLUMNS = ("record_id", "model_text", "language_type")
SENTIMENT_COLUMN = "sentiment"
BINARY_COLUMNS = (
    "joy",
    "affection",
    "amusement",
    "surprise",
    "sadness",
    "anger",
    "care_empathy",
    "fear",
    "disgust",
    "approval",
    "sarcasm",
)
CONFIDENCE_COLUMN = "confidence"
OPTIONAL_COLUMNS = ("other_emotion", "annotation_notes")
REQUIRED_COLUMNS = (
    *IDENTITY_COLUMNS,
    SENTIMENT_COLUMN,
    *BINARY_COLUMNS,
    CONFIDENCE_COLUMN,
    *OPTIONAL_COLUMNS,
)
SENTIMENT_LABELS = ("negative", "neutral", "positive", "mixed")
SENTIMENT_VALUES = {*SENTIMENT_LABELS, "uncertain"}
BINARY_VALUES = {"yes", "no", "uncertain"}
CONFIDENCE_VALUES = {"high", "medium", "low"}


def _normalize(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohen_kappa(left: pd.Series, right: pd.Series) -> float | None:
    """Calculate unweighted Cohen's kappa for paired categorical labels."""
    pairs = pd.DataFrame({"left": _normalize(left), "right": _normalize(right)}).dropna()
    if pairs.empty:
        return None
    observed = float((pairs["left"] == pairs["right"]).mean())
    labels = sorted(set(pairs["left"]) | set(pairs["right"]))
    expected = sum(
        float((pairs["left"] == label).mean())
        * float((pairs["right"] == label).mean())
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def classification_metrics(truth: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    """Return transparent multiclass metrics against adjudicated sentiment."""
    frame = pd.DataFrame({"truth": _normalize(truth), "prediction": _normalize(prediction)})
    frame = frame[frame["truth"].isin(SENTIMENT_LABELS)].copy()
    per_class: dict[str, dict[str, float | int]] = {}
    for label in SENTIMENT_LABELS:
        true_positive = int(((frame["truth"] == label) & (frame["prediction"] == label)).sum())
        false_positive = int(((frame["truth"] != label) & (frame["prediction"] == label)).sum())
        false_negative = int(((frame["truth"] == label) & (frame["prediction"] != label)).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": int((frame["truth"] == label).sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    active_labels = [label for label in SENTIMENT_LABELS if per_class[label]["support"]]
    confusion = {
        actual: {
            predicted: int(
                ((frame["truth"] == actual) & (frame["prediction"] == predicted)).sum()
            )
            for predicted in SENTIMENT_LABELS
        }
        for actual in SENTIMENT_LABELS
    }
    return {
        "evaluated_records": len(frame),
        "excluded_uncertain_records": int((~_normalize(truth).isin(SENTIMENT_LABELS)).sum()),
        "accuracy": float((frame["truth"] == frame["prediction"]).mean()),
        "macro_f1_active_human_classes": sum(
            float(per_class[label]["f1"]) for label in active_labels
        )
        / len(active_labels),
        "macro_f1_four_classes": sum(
            float(per_class[label]["f1"]) for label in SENTIMENT_LABELS
        )
        / len(SENTIMENT_LABELS),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def exact_mcnemar(
    truth: pd.Series, first_prediction: pd.Series, second_prediction: pd.Series
) -> dict[str, float | int]:
    """Two-sided exact McNemar comparison for paired classifier correctness."""
    frame = pd.DataFrame(
        {
            "truth": _normalize(truth),
            "first": _normalize(first_prediction),
            "second": _normalize(second_prediction),
        }
    )
    frame = frame[frame["truth"].isin(SENTIMENT_LABELS)]
    first_correct = frame["truth"] == frame["first"]
    second_correct = frame["truth"] == frame["second"]
    first_only = int((first_correct & ~second_correct).sum())
    second_only = int((~first_correct & second_correct).sum())
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(first_only, second_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "reactionfusion_correct_baseline_wrong": first_only,
        "reactionfusion_wrong_baseline_correct": second_only,
        "discordant_records": discordant,
        "two_sided_exact_p_value": p_value,
    }


def _load_and_validate_annotations(workbook_path: Path) -> dict[str, pd.DataFrame]:
    excel = pd.ExcelFile(workbook_path)
    missing_sheets = [sheet for sheet in SHEETS if sheet not in excel.sheet_names]
    if missing_sheets:
        raise ValueError(f"Missing annotation sheets: {missing_sheets}")

    frames: dict[str, pd.DataFrame] = {}
    for sheet in SHEETS:
        frame = pd.read_excel(workbook_path, sheet_name=sheet)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"{sheet} is missing columns: {missing_columns}")
        frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        if frame.empty:
            raise ValueError(f"{sheet} contains no annotation rows")
        if frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
            raise ValueError(f"{sheet} contains missing or duplicate record IDs")
        for column in (SENTIMENT_COLUMN, *BINARY_COLUMNS, CONFIDENCE_COLUMN):
            frame[column] = _normalize(frame[column])
            if frame[column].isna().any():
                raise ValueError(f"{sheet}.{column} contains blank required labels")
        invalid_sentiment = sorted(set(frame[SENTIMENT_COLUMN]) - SENTIMENT_VALUES)
        if invalid_sentiment:
            raise ValueError(f"{sheet}.sentiment has invalid labels: {invalid_sentiment}")
        for column in BINARY_COLUMNS:
            invalid = sorted(set(frame[column]) - BINARY_VALUES)
            if invalid:
                raise ValueError(f"{sheet}.{column} has invalid labels: {invalid}")
        invalid_confidence = sorted(set(frame[CONFIDENCE_COLUMN]) - CONFIDENCE_VALUES)
        if invalid_confidence:
            raise ValueError(f"{sheet}.confidence has invalid labels: {invalid_confidence}")
        frames[sheet] = frame

    reference = frames[SHEETS[0]][list(IDENTITY_COLUMNS)].reset_index(drop=True)
    for sheet in SHEETS[1:]:
        candidate = frames[sheet][list(IDENTITY_COLUMNS)].reset_index(drop=True)
        if not candidate.equals(reference):
            raise ValueError(f"Record IDs, text, or language do not align in {sheet}")
    return frames


def _agreement_report(annotator_1: pd.DataFrame, annotator_2: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for column in (SENTIMENT_COLUMN, *BINARY_COLUMNS, CONFIDENCE_COLUMN):
        left = _normalize(annotator_1[column])
        right = _normalize(annotator_2[column])
        report[column] = {
            "raw_agreement": float((left == right).mean()),
            "cohen_kappa": cohen_kappa(left, right),
            "disagreements": int((left != right).sum()),
        }
    return report


def _adjudication_audit(
    annotator_1: pd.DataFrame, annotator_2: pd.DataFrame, adjudication: pd.DataFrame
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for column in (SENTIMENT_COLUMN, *BINARY_COLUMNS, CONFIDENCE_COLUMN):
        first = _normalize(annotator_1[column])
        second = _normalize(annotator_2[column])
        final = _normalize(adjudication[column])
        report[column] = {
            "annotators_agreed": int((first == second).sum()),
            "final_matches_annotator_1": int((final == first).sum()),
            "final_matches_annotator_2": int((final == second).sum()),
            "final_matches_neither": int(((final != first) & (final != second)).sum()),
        }
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    agreement = report["inter_annotator_agreement"]["sentiment"]
    fusion = report["reactionfusion_vs_human"]
    baseline = report["filtered_baseline_vs_human"]
    paired = report["paired_accuracy_comparison"]
    distribution = report["human_sentiment_distribution"]
    return f"""# ReactionFusion v1 human-validation results

## Scope

The completed annotation workbook contains {report['records']} uncertainty-enriched
records selected by the ReactionFusion v1 annotation-candidate procedure. These
results describe performance on difficult/ambiguous cases and must **not** be
reported as an unbiased estimate for the full dataset.

Three adjudicated `uncertain` records are excluded from classifier metrics, leaving
{fusion['evaluated_records']} evaluated records.

## Annotation quality

- Sentiment raw agreement: {agreement['raw_agreement']:.3f}
- Sentiment Cohen's kappa: {agreement['cohen_kappa']:.3f}
- Sentiment disagreements: {agreement['disagreements']}
- Final sentiment distribution: negative={distribution.get('negative', 0)},
  neutral={distribution.get('neutral', 0)}, positive={distribution.get('positive', 0)},
  mixed={distribution.get('mixed', 0)}, uncertain={distribution.get('uncertain', 0)}

## Provisional comparison

| Method | Accuracy | Macro F1 (4 classes) |
|---|---:|---:|
| ReactionFusion v1 | {fusion['accuracy']:.3f} | {fusion['macro_f1_four_classes']:.3f} |
| Filtered-reaction baseline | {baseline['accuracy']:.3f} | {baseline['macro_f1_four_classes']:.3f} |

ReactionFusion v1 has higher macro F1 on this hard subset, while the filtered
baseline has higher accuracy. Neither method predicts the human `mixed` class in
the current configuration, so mixed-class recall is zero. The result does not yet
support a general claim that ReactionFusion is superior. The paired exact McNemar
test gives p={paired['two_sided_exact_p_value']:.3f}, so the accuracy difference is
not statistically significant at the 0.05 level. Use these errors to design v2 and
evaluate once on a separately frozen representative human test set.

## Next research action

Use the adjudicated development annotations to analyze failure patterns and tune a
versioned ReactionFusion v2 configuration. Do not overwrite v1. Freeze a separate,
representative human test sample before final model or algorithm comparison.
"""


def evaluate_human_annotations(
    workbook_path: Path, dataset_path: Path, output_dir: Path
) -> dict[str, Any]:
    frames = _load_and_validate_annotations(workbook_path)
    annotator_1 = frames["Annotator 1"]
    annotator_2 = frames["Annotator 2"]
    adjudication = frames["Adjudication"]

    dataset = pd.read_csv(dataset_path)
    prediction_columns = (
        "record_id",
        "sentiment_label",
        "baseline_filtered_label",
        "fusion_score",
        "label_confidence",
        "reaction_entropy",
        "is_ambiguous",
        "split",
    )
    missing = [column for column in prediction_columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing prediction columns: {missing}")
    merged = adjudication.merge(
        dataset.loc[:, prediction_columns], on="record_id", how="left", validate="one_to_one"
    )
    if merged["sentiment_label"].isna().any():
        raise ValueError("One or more adjudicated record IDs are absent from the dataset")

    human_distribution = dict(Counter(_normalize(adjudication[SENTIMENT_COLUMN])))
    report: dict[str, Any] = {
        "status": "completed_human_validation_on_uncertainty_enriched_subset",
        "records": len(adjudication),
        "annotation_workbook": str(workbook_path),
        "annotation_workbook_sha256": _sha256(workbook_path),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "sample_design": "uncertainty_enriched_not_representative",
        "human_sentiment_distribution": human_distribution,
        "inter_annotator_agreement": _agreement_report(annotator_1, annotator_2),
        "adjudication_audit": _adjudication_audit(annotator_1, annotator_2, adjudication),
        "reactionfusion_vs_human": classification_metrics(
            merged[SENTIMENT_COLUMN], merged["sentiment_label"]
        ),
        "filtered_baseline_vs_human": classification_metrics(
            merged[SENTIMENT_COLUMN], merged["baseline_filtered_label"]
        ),
        "paired_accuracy_comparison": exact_mcnemar(
            merged[SENTIMENT_COLUMN],
            merged["sentiment_label"],
            merged["baseline_filtered_label"],
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    human_labels = adjudication.rename(
        columns={
            column: f"human_{column}"
            for column in (SENTIMENT_COLUMN, *BINARY_COLUMNS, CONFIDENCE_COLUMN, *OPTIONAL_COLUMNS)
        }
    )
    human_labels.to_csv(output_dir / "adjudicated_human_labels.csv", index=False, encoding="utf-8-sig")
    merged_output = merged.rename(
        columns={
            column: f"human_{column}"
            for column in (SENTIMENT_COLUMN, *BINARY_COLUMNS, CONFIDENCE_COLUMN, *OPTIONAL_COLUMNS)
        }
    )
    merged_output.to_csv(output_dir / "human_validation_records.csv", index=False, encoding="utf-8-sig")
    (output_dir / "human_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/reactionfusion_v1/adjudication_completed.xlsx"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/releases/reactionfusion_v1/dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/releases/reactionfusion_v1/human_validation"),
    )
    args = parser.parse_args()
    report = evaluate_human_annotations(args.annotations, args.dataset, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
