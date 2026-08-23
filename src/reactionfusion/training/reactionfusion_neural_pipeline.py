"""Train, evaluate, serialize, and release neural ReactionFusion v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from reactionfusion.evaluation.human_validation import (
    _load_and_validate_annotations,
    classification_metrics,
)
from reactionfusion.labeling.reactionfusion_neural import (
    ReactionFusionNeuralModel,
    train_neural_model,
)
from reactionfusion.labeling.reactionfusion_v2 import BASE_FEATURE_NAMES, REACTIONS, feature_matrix
from reactionfusion.training.reactionfusion_v2_pipeline import (
    _apply_temperature,
    _paired_accuracy_test,
    assign_group_folds,
    select_abstention_threshold,
    select_temperature,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_grouped_oof(
    annotated: pd.DataFrame,
    features: np.ndarray,
    config: Mapping[str, Any],
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, int]]]:
    folds = assign_group_folds(
        annotated["sentiment"],
        annotated["text_hash"],
        n_splits=int(config["cross_validation_folds"]),
        seed=fold_seed,
    )
    sentiment = np.full((len(annotated), len(config["sentiment_classes"])), np.nan)
    emotion = np.full((len(annotated), len(config["emotion_targets"])), np.nan)
    summaries = []
    for fold in range(int(config["cross_validation_folds"])):
        validation = folds == fold
        training = ~validation
        train_groups = set(annotated.loc[training, "text_hash"])
        validation_groups = set(annotated.loc[validation, "text_hash"])
        if train_groups & validation_groups:
            raise AssertionError("Text group leaked across neural cross-validation folds")
        model = train_neural_model(
            features[training],
            annotated.loc[training].reset_index(drop=True),
            BASE_FEATURE_NAMES,
            config,
            metadata={"cross_validation_fold": fold, "fold_seed": fold_seed},
        )
        fold_sentiment, fold_emotion = model.predict_probability(features[validation])
        sentiment[validation] = fold_sentiment
        emotion[validation] = fold_emotion
        summaries.append(
            {
                "fold": fold,
                "training_records": int(training.sum()),
                "validation_records": int(validation.sum()),
                "training_groups": len(train_groups),
                "validation_groups": len(validation_groups),
            }
        )
    if np.isnan(sentiment).any() or np.isnan(emotion).any():
        raise AssertionError("Neural cross-validation did not predict every record")
    return sentiment, emotion, folds, summaries


def _evaluate_predictions(
    truth: pd.Series, probabilities: np.ndarray, classes: tuple[str, ...]
) -> tuple[pd.Series, dict[str, Any]]:
    predictions = pd.Series(
        np.asarray(classes)[probabilities.argmax(axis=1)], index=truth.index
    )
    return predictions, classification_metrics(truth, predictions)


def _cross_validate(
    annotated: pd.DataFrame,
    features: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, float, float]:
    classes = tuple(config["sentiment_classes"])
    repeat_probabilities = []
    repeat_emotions = []
    repeat_reports: dict[str, Any] = {}
    repeat_folds: dict[int, np.ndarray] = {}
    for seed_value in config["stability_fold_seeds"]:
        seed = int(seed_value)
        probabilities, emotions, folds, summaries = _run_grouped_oof(
            annotated, features, config, seed
        )
        prediction, metrics = _evaluate_predictions(
            annotated["sentiment"], probabilities, classes
        )
        repeat_probabilities.append(probabilities)
        repeat_emotions.append(emotions)
        repeat_folds[seed] = folds
        repeat_reports[str(seed)] = {
            "folds": summaries,
            "candidate_metrics": metrics,
            "candidate_distribution": dict(Counter(prediction)),
            "group_leakage_count": 0,
        }

    averaged = np.mean(repeat_probabilities, axis=0)
    averaged_emotions = np.mean(repeat_emotions, axis=0)
    temperature, temperature_losses = select_temperature(
        averaged,
        annotated["sentiment"],
        classes,
        [float(value) for value in config["temperature_candidates"]],
    )
    calibrated = _apply_temperature(averaged, temperature)
    threshold = select_abstention_threshold(calibrated, config)
    candidates, averaged_metrics = _evaluate_predictions(
        annotated["sentiment"], calibrated, classes
    )
    confidence = calibrated.max(axis=1)
    final_labels = np.where(confidence < threshold, "uncertain", candidates)
    final_metrics = classification_metrics(annotated["sentiment"], pd.Series(final_labels))

    v2_prediction = annotated["candidate_sentiment_label"].astype("string")
    v1_prediction = annotated["sentiment_label_v1"].astype("string")
    baseline_prediction = annotated["baseline_filtered_label"].astype("string")
    primary_seed = str(int(config["primary_fold_seed"]))
    repeat_accuracy = [
        float(report["candidate_metrics"]["accuracy"])
        for report in repeat_reports.values()
    ]
    repeat_macro_f1 = [
        float(report["candidate_metrics"]["macro_f1_four_classes"])
        for report in repeat_reports.values()
    ]
    report = {
        "evaluation_design": "repeated_five_fold_grouped_out_of_fold_development",
        "primary_fold_seed": int(config["primary_fold_seed"]),
        "repeat_reports": repeat_reports,
        "repeat_mean_accuracy": float(np.mean(repeat_accuracy)),
        "repeat_std_accuracy": float(np.std(repeat_accuracy)),
        "repeat_mean_macro_f1": float(np.mean(repeat_macro_f1)),
        "repeat_std_macro_f1": float(np.std(repeat_macro_f1)),
        "primary_candidate_metrics": repeat_reports[primary_seed]["candidate_metrics"],
        "averaged_oof_candidate_metrics": averaged_metrics,
        "averaged_oof_abstaining_metrics": final_metrics,
        "reactionfusion_v2_metrics": classification_metrics(
            annotated["sentiment"], v2_prediction
        ),
        "reactionfusion_v1_metrics": classification_metrics(
            annotated["sentiment"], v1_prediction
        ),
        "filtered_baseline_metrics": classification_metrics(
            annotated["sentiment"], baseline_prediction
        ),
        "paired_accuracy_neural_vs_v2": _paired_accuracy_test(
            annotated["sentiment"], candidates, v2_prediction
        ),
        "temperature": temperature,
        "temperature_log_losses": temperature_losses,
        "abstention_threshold": threshold,
        "target_coverage": float(config["target_coverage"]),
        "observed_coverage": float((confidence >= threshold).mean()),
        "group_leakage_count": 0,
    }

    oof = annotated[["record_id", "text_hash", "sentiment"]].copy()
    for seed, folds in repeat_folds.items():
        oof[f"cv_fold_seed_{seed}"] = folds
    oof["candidate_sentiment_label"] = candidates.to_numpy()
    oof["sentiment_label"] = final_labels
    oof["label_confidence"] = confidence
    oof["is_uncertain"] = confidence < threshold
    for index, label in enumerate(classes):
        oof[f"sentiment_probability_{label}"] = calibrated[:, index]
    for index, emotion in enumerate(config["emotion_targets"]):
        oof[f"emotion_probability_{emotion}"] = averaged_emotions[:, index]
    return report, oof, temperature, threshold


def _rename_v2_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    generic = (
        "sentiment_label",
        "candidate_sentiment_label",
        "label_confidence",
        "is_uncertain",
        "is_ambiguous",
        "reaction_entropy",
        "mixed_evidence",
        "decision_reason",
        "prediction_source",
        "label_version",
    )
    probability_columns = [
        column
        for column in dataset.columns
        if column.startswith("sentiment_probability_")
        or column.startswith("emotion_probability_")
    ]
    rename = {column: f"{column}_v2" for column in (*generic, *probability_columns)}
    return dataset.rename(columns={key: value for key, value in rename.items() if key in dataset})


def _apply_predictions(
    dataset: pd.DataFrame,
    model: ReactionFusionNeuralModel,
) -> pd.DataFrame:
    release = _rename_v2_columns(dataset.copy())
    features = feature_matrix(release, model.smoothing_alpha)
    sentiment, emotion = model.predict_probability(features)
    classes = model.sentiment_classes
    candidate_ids = sentiment.argmax(axis=1)
    candidates = np.asarray(classes)[candidate_ids]
    confidence = sentiment.max(axis=1)
    uncertain = confidence < model.abstention_threshold
    release["candidate_sentiment_label"] = candidates
    release["sentiment_label"] = np.where(uncertain, "uncertain", candidates)
    release["label_confidence"] = confidence
    release["is_uncertain"] = uncertain
    release["is_ambiguous"] = uncertain
    for index, label in enumerate(classes):
        release[f"sentiment_probability_{label}"] = sentiment[:, index]
    for index, target in enumerate(model.emotion_targets):
        release[f"emotion_probability_{target}"] = emotion[:, index]
    release["decision_reason"] = np.where(
        uncertain,
        "neural ensemble confidence below calibrated threshold",
        "neural ensemble candidate accepted",
    )
    release["prediction_source"] = "final_neural_ensemble"
    release["label_version"] = model.version
    return release


def _apply_oof_overrides(release: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    indexed = oof.set_index("record_id")
    mask = release["record_id"].isin(indexed.index)
    target_rows = release.loc[mask, "record_id"]
    columns = [
        "candidate_sentiment_label",
        "sentiment_label",
        "label_confidence",
        "is_uncertain",
        *(
            f"sentiment_probability_{label}"
            for label in ("negative", "neutral", "positive", "mixed")
        ),
        *(column for column in indexed.columns if column.startswith("emotion_probability_")),
    ]
    columns = list(dict.fromkeys(columns))
    for column in columns:
        if column in indexed:
            release.loc[mask, column] = target_rows.map(indexed[column]).to_numpy()
    release.loc[mask, "is_ambiguous"] = release.loc[mask, "is_uncertain"]
    release.loc[mask, "decision_reason"] = np.where(
        release.loc[mask, "is_uncertain"],
        "repeated grouped OOF confidence below calibrated threshold",
        "repeated grouped OOF neural candidate accepted",
    )
    release.loc[mask, "prediction_source"] = "repeated_grouped_out_of_fold"
    return release


def _dataset_card(report: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    primary = report["primary_candidate_metrics"]
    averaged = report["averaged_oof_candidate_metrics"]
    v2 = report["reactionfusion_v2_metrics"]
    p_value = report["paired_accuracy_neural_vs_v2"]["two_sided_exact_mcnemar_p_value"]
    primary_accuracy = primary["accuracy"]
    primary_f1 = primary["macro_f1_four_classes"]
    averaged_accuracy = averaged["accuracy"]
    averaged_f1 = averaged["macro_f1_four_classes"]
    v2_accuracy = v2["accuracy"]
    v2_f1 = v2["macro_f1_four_classes"]
    return f"""# ReactionFusion neural v3 development dataset card

## Status

This is an **experimental development release**, not a final benchmark. The neural
architecture and its hyperparameters were developed using the same 120-record
uncertainty-enriched annotation set used for evaluation.

## Model

- Inputs: reaction counts and 27 derived distribution features; no post text.
- Five-member ensemble with one 16-unit tanh hidden layer per member.
- Multi-task outputs: four-class sentiment and eleven emotion/stance probabilities.
- Class weighting, label smoothing, L2 regularization, feature noise, temperature
  calibration, and confidence-based abstention.
- Repeated grouped out-of-fold labels for every human-calibration record.

## Development results

| Evaluation | Accuracy | Four-class macro F1 |
|---|---:|---:|
| Neural primary grouped five-fold | {primary_accuracy:.3f} | {primary_f1:.3f} |
| Neural averaged repeated OOF | {averaged_accuracy:.3f} | {averaged_f1:.3f} |
| ReactionFusion v2 | {v2_accuracy:.3f} | {v2_f1:.3f} |

Across fold seeds, mean accuracy is {report['repeat_mean_accuracy']:.3f} and mean
macro F1 is {report['repeat_mean_macro_f1']:.3f}. The exact paired McNemar p-value
for averaged neural predictions versus v2 is {p_value:.3f}.

## Generated release

- Records: {quality['records']}
- Deduplicated records: {quality['deduplicated_records']}
- Labels: {json.dumps(quality['label_distribution'], ensure_ascii=False)}

## Research limitation

The neural model provides only a small and fold-sensitive development improvement.
More representative human annotations—especially mixed, neutral, and rare emotion
examples—are required before claiming that the neural architecture outperforms v2.
Do not use the existing 120 records as the final research test set.
"""


def train_and_release_neural(
    dataset_path: Path,
    annotation_path: Path,
    config_path: Path,
    release_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = pd.read_csv(dataset_path, low_memory=False)
    required = ["record_id", "text_hash", "split", *(f"{name}_count" for name in REACTIONS)]
    missing = [column for column in required if column not in dataset]
    if missing:
        raise ValueError(f"V2 dataset is missing neural inputs: {missing}")
    annotations = _load_and_validate_annotations(annotation_path)["Adjudication"]
    annotated = annotations.merge(dataset, on="record_id", how="left", validate="one_to_one")
    if annotated[[f"{name}_count" for name in REACTIONS]].isna().any().any():
        raise ValueError("Annotated records are missing reaction counts")

    features = feature_matrix(annotated, float(config["smoothing_alpha"]))
    report, oof, temperature, threshold = _cross_validate(annotated, features, config)
    model = train_neural_model(
        features,
        annotated,
        BASE_FEATURE_NAMES,
        config,
        metadata={
            "annotation_workbook_sha256": _sha256(annotation_path),
            "v2_dataset_sha256": _sha256(dataset_path),
            "training_records": len(annotated),
            "sample_design": "uncertainty_enriched_development",
        },
    )
    model = replace(model, temperature=temperature, abstention_threshold=threshold)
    release = _apply_oof_overrides(_apply_predictions(dataset, model), oof)
    deduplicated = release.sort_values(
        ["text_hash", "total_reactions"], ascending=[True, False]
    ).drop_duplicates("text_hash")

    release_dir.mkdir(parents=True, exist_ok=True)
    release.to_csv(release_dir / "dataset.csv", index=False, encoding="utf-8-sig")
    deduplicated.to_csv(
        release_dir / "dataset_deduplicated.csv", index=False, encoding="utf-8-sig"
    )
    for split in ("train", "validation", "test"):
        release[release["split"] == split].to_csv(
            release_dir / f"{split}.csv", index=False, encoding="utf-8-sig"
        )
    oof.to_csv(release_dir / "human_validation_oof.csv", index=False, encoding="utf-8-sig")
    (release_dir / "model.json").write_text(
        json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (release_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (release_dir / "cross_validation_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    quality = {
        "records": len(release),
        "deduplicated_records": len(deduplicated),
        "label_distribution": dict(Counter(release["sentiment_label"])),
        "candidate_distribution": dict(Counter(release["candidate_sentiment_label"])),
        "uncertain_records": int(release["is_uncertain"].sum()),
        "confident_records": int((~release["is_uncertain"]).sum()),
        "split_distribution": release["split"].value_counts().sort_index().to_dict(),
        "prediction_source_distribution": dict(Counter(release["prediction_source"])),
        "version": model.version,
        "source_v2_dataset_sha256": _sha256(dataset_path),
        "annotation_workbook_sha256": _sha256(annotation_path),
    }
    (release_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (release_dir / "DATASET_CARD.md").write_text(
        _dataset_card(report, quality), encoding="utf-8"
    )
    return {"cross_validation": report, "quality": quality}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/releases/reactionfusion_v2/dataset.csv")
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/reactionfusion_v1/adjudication_completed.xlsx"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/labeling/reactionfusion_neural_v3.json"),
    )
    parser.add_argument(
        "--release-dir", type=Path, default=Path("data/releases/reactionfusion_neural_v3")
    )
    args = parser.parse_args()
    report = train_and_release_neural(
        args.dataset, args.annotations, args.config, args.release_dir
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
