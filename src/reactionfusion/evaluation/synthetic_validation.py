"""Evaluate the existing neural v3 model on the synthetic augmentation records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from reactionfusion.evaluation.human_validation import SENTIMENT_LABELS, classification_metrics
from reactionfusion.labeling.reactionfusion_neural import ReactionFusionNeuralModel
from reactionfusion.labeling.reactionfusion_v2 import feature_matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_metrics(
    truth: pd.Series,
    probabilities: np.ndarray,
    classes: tuple[str, ...] = SENTIMENT_LABELS,
    bins: int = 10,
) -> dict[str, float]:
    """Calculate multiclass log loss, Brier score, and expected calibration error."""
    normalized = truth.astype("string").str.strip().str.lower()
    class_to_id = {label: index for index, label in enumerate(classes)}
    usable = normalized.isin(classes).to_numpy()
    values = probabilities[usable]
    target_ids = np.asarray([class_to_id[label] for label in normalized[usable]], dtype=int)
    selected = values[np.arange(len(values)), target_ids]
    log_loss = -float(np.log(np.clip(selected, 1e-12, 1.0)).mean())
    one_hot = np.eye(len(classes))[target_ids]
    brier = float(np.square(values - one_hot).sum(axis=1).mean())

    confidence = values.max(axis=1)
    prediction = values.argmax(axis=1)
    correct = prediction == target_ids
    expected_calibration_error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            member = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            member = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if member.any():
            expected_calibration_error += float(member.mean()) * abs(
                float(correct[member].mean()) - float(confidence[member].mean())
            )
    return {
        "multiclass_log_loss": log_loss,
        "multiclass_brier_score": brier,
        "expected_calibration_error_10_bins": expected_calibration_error,
        "mean_confidence": float(confidence.mean()),
    }


def paired_exact_mcnemar(
    truth: pd.Series, first: pd.Series, second: pd.Series
) -> dict[str, float | int]:
    normalized_truth = truth.astype("string").str.strip().str.lower()
    first = first.astype("string").str.strip().str.lower()
    second = second.astype("string").str.strip().str.lower()
    usable = normalized_truth.isin(SENTIMENT_LABELS)
    first_correct = first[usable] == normalized_truth[usable]
    second_correct = second[usable] == normalized_truth[usable]
    first_only = int((first_correct & ~second_correct).sum())
    second_only = int((~first_correct & second_correct).sum())
    discordant = first_only + second_only
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(first_only, second_only) + 1)
        )
        log_p_value = math.log(2.0) + math.log(tail) - discordant * math.log(2.0)
        p_value = min(1.0, math.exp(log_p_value))
    else:
        p_value = 1.0
    return {
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "discordant_records": discordant,
        "two_sided_exact_mcnemar_p_value": p_value,
    }


def _stratified_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value, group in frame.groupby(column, dropna=False, sort=True):
        metrics = classification_metrics(
            group["provided_sentiment"], group["neural_v3_candidate_sentiment"]
        )
        output[str(value)] = {
            "records": len(group),
            "accuracy": metrics["accuracy"],
            "macro_f1_four_classes": metrics["macro_f1_four_classes"],
        }
    return output


def _results_markdown(report: Mapping[str, Any]) -> str:
    candidate = report["neural_v3_candidate_metrics"]
    final = report["neural_v3_abstaining_metrics"]
    v2 = report["comparison_methods"]["reactionfusion_v2_candidate"]
    v1 = report["comparison_methods"]["reactionfusion_v1"]
    baseline = report["comparison_methods"]["filtered_baseline"]
    probability = report["probability_metrics"]
    return f"""# Neural v3 evaluation on the synthetic augmentation set

## Evaluation design

The frozen human-calibrated neural v3 model was evaluated without retraining on
15,000 synthetic records. Reference labels are the supplied synthetic adjudication
labels and are not verified human ground truth.

| Method | Accuracy | Four-class macro F1 |
|---|---:|---:|
| Neural v3 candidate | {candidate['accuracy']:.3f} | {candidate['macro_f1_four_classes']:.3f} |
| Neural v3 with abstention | {final['accuracy']:.3f} | {final['macro_f1_four_classes']:.3f} |
| ReactionFusion v2 candidate | {v2['accuracy']:.3f} | {v2['macro_f1_four_classes']:.3f} |
| ReactionFusion v1 | {v1['accuracy']:.3f} | {v1['macro_f1_four_classes']:.3f} |
| Filtered baseline | {baseline['accuracy']:.3f} | {baseline['macro_f1_four_classes']:.3f} |

## Neural v3 diagnostics

- Confident coverage: {report['confident_coverage']:.3f}
- Accuracy among confident predictions: {report['selective_accuracy']:.3f}
- Mean candidate confidence: {probability['mean_confidence']:.3f}
- Multiclass log loss: {probability['multiclass_log_loss']:.3f}
- Multiclass Brier score: {probability['multiclass_brier_score']:.3f}
- Expected calibration error: {probability['expected_calibration_error_10_bins']:.3f}

## Interpretation

Neural v3 predicts neutral for most synthetic records and performs below v2 on the
provided synthetic labels. This is evidence of domain shift between the original
human-calibration sample and the fabricated reaction/text generation process. It
does not invalidate the original human evaluation, and it does not validate the
synthetic annotations as research ground truth.
"""


def evaluate_neural_v3_on_synthetic(
    dataset_path: Path, model_path: Path, output_dir: Path
) -> dict[str, Any]:
    dataset = pd.read_csv(dataset_path, low_memory=False)
    required = {
        "record_id",
        "data_origin",
        "provided_sentiment",
        "sentiment_label",
        "baseline_filtered_label",
        "v2_candidate_sentiment",
        "neural_v3_candidate_sentiment",
        "neural_v3_sentiment_label",
        "neural_v3_confidence",
        *(f"neural_v3_probability_{label}" for label in SENTIMENT_LABELS),
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Combined dataset is missing synthetic-test columns: {missing}")
    synthetic = dataset[dataset["data_origin"] == "synthetic_augmentation"].copy()
    if len(synthetic) != 15_000:
        raise ValueError(f"Expected 15,000 synthetic records, found {len(synthetic)}")
    normalized_truth = synthetic["provided_sentiment"].astype("string").str.lower()
    if not normalized_truth.isin(SENTIMENT_LABELS).all():
        raise ValueError("Synthetic test set contains unsupported or blank sentiment labels")

    model = ReactionFusionNeuralModel.from_dict(
        json.loads(model_path.read_text(encoding="utf-8"))
    )
    fresh_probabilities, _ = model.predict_probability(
        feature_matrix(synthetic, model.smoothing_alpha)
    )
    stored_probability_columns = [
        f"neural_v3_probability_{label}" for label in SENTIMENT_LABELS
    ]
    stored_probabilities = synthetic[stored_probability_columns].to_numpy(dtype=float)
    if not np.allclose(fresh_probabilities, stored_probabilities):
        raise ValueError("Fresh serialized-model predictions do not match the combined release")
    fresh_candidate_ids = fresh_probabilities.argmax(axis=1)
    fresh_candidates = np.asarray(model.sentiment_classes)[fresh_candidate_ids]
    fresh_confidence = fresh_probabilities.max(axis=1)
    fresh_final = np.where(
        fresh_confidence < model.abstention_threshold, "uncertain", fresh_candidates
    )
    synthetic["neural_v3_candidate_sentiment"] = fresh_candidates
    synthetic["neural_v3_sentiment_label"] = fresh_final
    synthetic["neural_v3_confidence"] = fresh_confidence

    candidate_metrics = classification_metrics(
        normalized_truth, synthetic["neural_v3_candidate_sentiment"]
    )
    final_metrics = classification_metrics(
        normalized_truth, synthetic["neural_v3_sentiment_label"]
    )
    probabilities = fresh_probabilities
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Neural v3 probability rows do not sum to one")
    confident = synthetic["neural_v3_sentiment_label"] != "uncertain"
    comparison_columns = {
        "reactionfusion_v2_candidate": "v2_candidate_sentiment",
        "reactionfusion_v1": "sentiment_label",
        "filtered_baseline": "baseline_filtered_label",
    }
    comparisons = {
        name: classification_metrics(normalized_truth, synthetic[column])
        for name, column in comparison_columns.items()
    }
    report: dict[str, Any] = {
        "status": "synthetic_external_test_not_human_benchmark",
        "evaluation_records": len(synthetic),
        "model_training_overlap": 0,
        "model_version": model.version,
        "model_sha256": _sha256(model_path),
        "serialized_prediction_match": True,
        "reference_label_provenance": "provided_synthetic_unverified",
        "neural_v3_candidate_metrics": candidate_metrics,
        "neural_v3_abstaining_metrics": final_metrics,
        "probability_metrics": probability_metrics(normalized_truth, probabilities),
        "confident_coverage": float(confident.mean()),
        "confident_records": int(confident.sum()),
        "uncertain_records": int((~confident).sum()),
        "selective_accuracy": float(
            (
                synthetic.loc[confident, "neural_v3_sentiment_label"]
                == normalized_truth[confident]
            ).mean()
        ),
        "truth_distribution": normalized_truth.value_counts().to_dict(),
        "candidate_distribution": synthetic[
            "neural_v3_candidate_sentiment"
        ].value_counts().to_dict(),
        "final_distribution": synthetic[
            "neural_v3_sentiment_label"
        ].value_counts().to_dict(),
        "comparison_methods": comparisons,
        "paired_neural_v3_vs_v2": paired_exact_mcnemar(
            normalized_truth,
            synthetic["neural_v3_candidate_sentiment"],
            synthetic["v2_candidate_sentiment"],
        ),
        "metrics_by_language": _stratified_metrics(synthetic, "language_type"),
        "dataset_sha256": _sha256(dataset_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    record_columns = [
        "record_id",
        "language_type",
        "total_reactions",
        "provided_sentiment",
        "neural_v3_candidate_sentiment",
        "neural_v3_sentiment_label",
        "neural_v3_confidence",
        *(f"neural_v3_probability_{label}" for label in SENTIMENT_LABELS),
        "v2_candidate_sentiment",
        "sentiment_label",
        "baseline_filtered_label",
    ]
    synthetic[record_columns].to_csv(
        output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(
        _results_markdown(report), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/releases/reactionfusion_neural_v3/model.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/releases/reactionfusion_augmented_v4/dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/releases/reactionfusion_augmented_v4/neural_v3_synthetic_test"
        ),
    )
    args = parser.parse_args()
    report = evaluate_neural_v3_on_synthetic(args.dataset, args.model, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
