"""Train provenance-weighted neural ReactionFusion v5 on legacy and synthetic data."""

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
    BINARY_COLUMNS,
    classification_metrics,
)
from reactionfusion.evaluation.synthetic_validation import paired_exact_mcnemar
from reactionfusion.labeling.reactionfusion_neural import (
    ReactionFusionNeuralModel,
    train_neural_model,
)
from reactionfusion.labeling.reactionfusion_v2 import BASE_FEATURE_NAMES, REACTIONS, feature_matrix
from reactionfusion.training.reactionfusion_v2_pipeline import (
    _apply_temperature,
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


def _annotation_frame(records: pd.DataFrame, weight: float) -> pd.DataFrame:
    columns = ["record_id", "provided_sentiment", *(f"provided_{c}" for c in BINARY_COLUMNS)]
    missing = [column for column in columns if column not in records]
    if missing:
        raise ValueError(f"Combined records are missing annotation columns: {missing}")
    rename = {
        "provided_sentiment": "sentiment",
        **{f"provided_{column}": column for column in BINARY_COLUMNS},
    }
    annotations = records[columns].rename(columns=rename).reset_index(drop=True)
    annotations["sample_weight"] = float(weight)
    return annotations


def _train_joint_model(
    synthetic_features: np.ndarray,
    synthetic_annotations: pd.DataFrame,
    human_features: np.ndarray,
    human_annotations: pd.DataFrame,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> ReactionFusionNeuralModel:
    features = np.vstack([synthetic_features, human_features])
    annotations = pd.concat(
        [synthetic_annotations, human_annotations], ignore_index=True
    )
    return train_neural_model(
        features, annotations, BASE_FEATURE_NAMES, config, metadata=metadata
    )


def _run_human_oof(
    synthetic_features: np.ndarray,
    synthetic_annotations: pd.DataFrame,
    human: pd.DataFrame,
    human_features: np.ndarray,
    human_annotations: pd.DataFrame,
    config: Mapping[str, Any],
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, int]]]:
    folds = assign_group_folds(
        human_annotations["sentiment"],
        human["text_hash"].reset_index(drop=True),
        n_splits=int(config["cross_validation_folds"]),
        seed=fold_seed,
    )
    classes = tuple(config["sentiment_classes"])
    emotions = tuple(config["emotion_targets"])
    probabilities = np.full((len(human), len(classes)), np.nan)
    emotion_probabilities = np.full((len(human), len(emotions)), np.nan)
    summaries = []
    for fold in range(int(config["cross_validation_folds"])):
        validation = folds == fold
        training = ~validation
        train_groups = set(human.loc[training, "text_hash"])
        validation_groups = set(human.loc[validation, "text_hash"])
        if train_groups & validation_groups:
            raise AssertionError("Human text group leaked across combined-model folds")
        model = _train_joint_model(
            synthetic_features,
            synthetic_annotations,
            human_features[training],
            human_annotations.loc[training].reset_index(drop=True),
            config,
            metadata={"fold_seed": fold_seed, "fold": fold},
        )
        fold_sentiment, fold_emotions = model.predict_probability(
            human_features[validation]
        )
        probabilities[validation] = fold_sentiment
        emotion_probabilities[validation] = fold_emotions
        summaries.append(
            {
                "fold": fold,
                "synthetic_training_records": len(synthetic_annotations),
                "human_training_records": int(training.sum()),
                "human_validation_records": int(validation.sum()),
                "human_training_groups": len(train_groups),
                "human_validation_groups": len(validation_groups),
            }
        )
    if np.isnan(probabilities).any() or np.isnan(emotion_probabilities).any():
        raise AssertionError("Combined cross-validation did not predict every human record")
    return probabilities, emotion_probabilities, folds, summaries


def _evaluate_human_oof(
    synthetic_features: np.ndarray,
    synthetic_annotations: pd.DataFrame,
    human: pd.DataFrame,
    human_features: np.ndarray,
    human_annotations: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, float, float]:
    classes = tuple(config["sentiment_classes"])
    repeat_probabilities = []
    repeat_emotions = []
    repeat_folds: dict[int, np.ndarray] = {}
    repeat_reports: dict[str, Any] = {}
    for seed_value in config["stability_fold_seeds"]:
        seed = int(seed_value)
        probabilities, emotions, folds, summaries = _run_human_oof(
            synthetic_features,
            synthetic_annotations,
            human,
            human_features,
            human_annotations,
            config,
            seed,
        )
        predictions = pd.Series(np.asarray(classes)[probabilities.argmax(axis=1)])
        metrics = classification_metrics(human_annotations["sentiment"], predictions)
        repeat_probabilities.append(probabilities)
        repeat_emotions.append(emotions)
        repeat_folds[seed] = folds
        repeat_reports[str(seed)] = {"folds": summaries, "candidate_metrics": metrics}

    averaged = np.mean(repeat_probabilities, axis=0)
    averaged_emotions = np.mean(repeat_emotions, axis=0)
    temperature, temperature_losses = select_temperature(
        averaged,
        human_annotations["sentiment"],
        classes,
        [float(value) for value in config["temperature_candidates"]],
    )
    calibrated = _apply_temperature(averaged, temperature)
    threshold = select_abstention_threshold(calibrated, config)
    candidates = pd.Series(np.asarray(classes)[calibrated.argmax(axis=1)])
    confidence = calibrated.max(axis=1)
    final = pd.Series(np.where(confidence < threshold, "uncertain", candidates))
    candidate_metrics = classification_metrics(human_annotations["sentiment"], candidates)
    final_metrics = classification_metrics(human_annotations["sentiment"], final)
    old_v3 = human["neural_v3_candidate_sentiment"].reset_index(drop=True)
    v2 = human["v2_candidate_sentiment"].reset_index(drop=True)

    accuracies = [
        float(item["candidate_metrics"]["accuracy"]) for item in repeat_reports.values()
    ]
    macro_f1_values = [
        float(item["candidate_metrics"]["macro_f1_four_classes"])
        for item in repeat_reports.values()
    ]
    primary = repeat_reports[str(int(config["primary_fold_seed"]))]["candidate_metrics"]
    report = {
        "evaluation_design": (
            "repeated_grouped_human_oof_with_all_synthetic_records_in_training_only"
        ),
        "synthetic_training_records_per_fold": len(synthetic_annotations),
        "human_evaluation_records": len(human),
        "human_group_leakage_count": 0,
        "cross_origin_text_overlap": 0,
        "repeat_reports": repeat_reports,
        "repeat_mean_accuracy": float(np.mean(accuracies)),
        "repeat_std_accuracy": float(np.std(accuracies)),
        "repeat_mean_macro_f1": float(np.mean(macro_f1_values)),
        "repeat_std_macro_f1": float(np.std(macro_f1_values)),
        "primary_candidate_metrics": primary,
        "averaged_oof_candidate_metrics": candidate_metrics,
        "averaged_oof_abstaining_metrics": final_metrics,
        "neural_v3_metrics": classification_metrics(
            human_annotations["sentiment"], old_v3
        ),
        "reactionfusion_v2_metrics": classification_metrics(
            human_annotations["sentiment"], v2
        ),
        "paired_combined_v5_vs_neural_v3": paired_exact_mcnemar(
            human_annotations["sentiment"], candidates, old_v3
        ),
        "temperature": temperature,
        "temperature_log_losses": temperature_losses,
        "abstention_threshold": threshold,
        "observed_coverage": float((confidence >= threshold).mean()),
    }

    oof = human[["record_id", "text_hash", "provided_sentiment"]].reset_index(drop=True)
    for seed, folds in repeat_folds.items():
        oof[f"cv_fold_seed_{seed}"] = folds
    oof["candidate_sentiment_label"] = candidates
    oof["sentiment_label"] = final
    oof["label_confidence"] = confidence
    oof["is_uncertain"] = confidence < threshold
    for index, label in enumerate(classes):
        oof[f"sentiment_probability_{label}"] = calibrated[:, index]
    for index, emotion in enumerate(config["emotion_targets"]):
        oof[f"emotion_probability_{emotion}"] = averaged_emotions[:, index]
    return report, oof, temperature, threshold


def _prediction_frame(
    dataset: pd.DataFrame, model: ReactionFusionNeuralModel
) -> pd.DataFrame:
    sentiment, emotions = model.predict_probability(
        feature_matrix(dataset, model.smoothing_alpha)
    )
    candidates = np.asarray(model.sentiment_classes)[sentiment.argmax(axis=1)]
    confidence = sentiment.max(axis=1)
    uncertain = confidence < model.abstention_threshold
    output = pd.DataFrame(index=dataset.index)
    output["candidate_sentiment_label"] = candidates
    output["sentiment_label"] = np.where(uncertain, "uncertain", candidates)
    output["label_confidence"] = confidence
    output["is_uncertain"] = uncertain
    for index, label in enumerate(model.sentiment_classes):
        output[f"sentiment_probability_{label}"] = sentiment[:, index]
    for index, emotion in enumerate(model.emotion_targets):
        output[f"emotion_probability_{emotion}"] = emotions[:, index]
    output["prediction_source"] = "final_provenance_weighted_model"
    output["label_version"] = model.version
    return output


def _override_human_oof(
    predictions: pd.DataFrame, dataset: pd.DataFrame, oof: pd.DataFrame
) -> None:
    indexed = oof.set_index("record_id")
    mask = dataset["record_id"].isin(indexed.index)
    record_ids = dataset.loc[mask, "record_id"]
    columns = [
        "candidate_sentiment_label",
        "sentiment_label",
        "label_confidence",
        "is_uncertain",
        *(column for column in oof if column.startswith("sentiment_probability_")),
        *(column for column in oof if column.startswith("emotion_probability_")),
    ]
    for column in columns:
        predictions.loc[mask, column] = record_ids.map(indexed[column]).to_numpy()
    predictions.loc[mask, "prediction_source"] = "repeated_grouped_human_oof"


def _dataset_card(report: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    primary = report["primary_candidate_metrics"]
    averaged = report["averaged_oof_candidate_metrics"]
    old = report["neural_v3_metrics"]
    paired = report["paired_combined_v5_vs_neural_v3"]
    primary_accuracy = primary["accuracy"]
    primary_f1 = primary["macro_f1_four_classes"]
    averaged_accuracy = averaged["accuracy"]
    averaged_f1 = averaged["macro_f1_four_classes"]
    old_accuracy = old["accuracy"]
    old_f1 = old["macro_f1_four_classes"]
    return f"""# ReactionFusion neural combined v5 dataset card

## Training design

All 15,000 supplied synthetic annotations and all 120 legacy human annotations
participate in joint multi-task neural training. Provenance weights are 0.0001 for
each synthetic record and 1.0 for each human record because the earlier synthetic
transfer experiment failed on human labels.

Human evaluation remains leakage-safe: each human record is predicted out-of-fold,
while all synthetic records are training-only augmentation. There is zero
normalized-text overlap between the two origins.

## Human development results

| Model/evaluation | Accuracy | Four-class macro F1 |
|---|---:|---:|
| Combined v5 primary five-fold | {primary_accuracy:.3f} | {primary_f1:.3f} |
| Combined v5 averaged repeated OOF | {averaged_accuracy:.3f} | {averaged_f1:.3f} |
| Human-only neural v3 | {old_accuracy:.3f} | {old_f1:.3f} |

The exact paired McNemar p-value for combined v5 versus neural v3 is
{paired['two_sided_exact_mcnemar_p_value']:.3f}. These are development statistics
on the original uncertainty-enriched human sample, not final benchmark results.

## Release

- Records: {quality['records']}
- Human OOF records: {quality['human_oof_records']}
- Synthetic records used in training: {quality['synthetic_training_records']}
- Uncertain output records: {quality['uncertain_records']}

Synthetic labels remain unverified and must not be described as human ground truth.
"""


def train_and_release_combined(
    dataset_path: Path,
    config_path: Path,
    release_dir: Path,
) -> dict[str, Any]:
    dataset = pd.read_csv(dataset_path, low_memory=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = [
        "record_id",
        "text_hash",
        "split",
        "annotation_provenance",
        "provided_sentiment",
        *(f"{reaction}_count" for reaction in REACTIONS),
    ]
    missing = [column for column in required if column not in dataset]
    if missing:
        raise ValueError(f"Combined dataset is missing v5 inputs: {missing}")
    synthetic = dataset[
        dataset["annotation_provenance"] == "provided_synthetic_unverified"
    ].reset_index(drop=True)
    human = dataset[
        dataset["annotation_provenance"] == "human_adjudicated_development"
    ].reset_index(drop=True)
    if len(synthetic) != 15_000 or len(human) != 120:
        raise ValueError("Combined training requires 15,000 synthetic and 120 human records")
    if set(synthetic["text_hash"]) & set(human["text_hash"]):
        raise ValueError("Synthetic and human annotation text groups overlap")

    synthetic_annotations = _annotation_frame(
        synthetic, float(config["synthetic_sample_weight"])
    )
    human_annotations = _annotation_frame(
        human, float(config["human_sample_weight"])
    )
    synthetic_features = feature_matrix(synthetic, float(config["smoothing_alpha"]))
    human_features = feature_matrix(human, float(config["smoothing_alpha"]))
    report, oof, temperature, threshold = _evaluate_human_oof(
        synthetic_features,
        synthetic_annotations,
        human,
        human_features,
        human_annotations,
        config,
    )
    model = _train_joint_model(
        synthetic_features,
        synthetic_annotations,
        human_features,
        human_annotations,
        config,
        metadata={
            "combined_dataset_sha256": _sha256(dataset_path),
            "synthetic_training_records": len(synthetic),
            "human_joint_training_records": len(human),
            "synthetic_sample_weight": float(config["synthetic_sample_weight"]),
            "human_sample_weight": float(config["human_sample_weight"]),
        },
    )
    model = replace(model, temperature=temperature, abstention_threshold=threshold)
    predictions = _prediction_frame(dataset, model)
    _override_human_oof(predictions, dataset, oof)

    core_columns = [
        "record_id",
        "model_text",
        "language_type",
        "text_hash",
        "split",
        *(f"{reaction}_count" for reaction in REACTIONS),
        "total_reactions",
        "data_origin",
        "text_provenance",
        "reaction_provenance",
        "annotation_provenance",
        "provided_sentiment",
        "neural_v3_candidate_sentiment",
        "neural_v3_sentiment_label",
        "neural_v3_confidence",
    ]
    release = pd.concat(
        [dataset[core_columns].reset_index(drop=True), predictions.reset_index(drop=True)],
        axis=1,
    )
    release_dir.mkdir(parents=True, exist_ok=True)
    release.to_csv(release_dir / "dataset.csv", index=False, encoding="utf-8-sig")
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
        "human_oof_records": int(
            (release["prediction_source"] == "repeated_grouped_human_oof").sum()
        ),
        "synthetic_training_records": len(synthetic),
        "uncertain_records": int(release["is_uncertain"].sum()),
        "label_distribution": dict(Counter(release["sentiment_label"])),
        "prediction_source_distribution": dict(Counter(release["prediction_source"])),
        "cross_split_text_group_leakage": int(
            (release.groupby("text_hash")["split"].nunique() > 1).sum()
        ),
        "version": model.version,
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
        "--dataset",
        type=Path,
        default=Path("data/releases/reactionfusion_augmented_v4/dataset.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/labeling/reactionfusion_neural_combined_v5.json"),
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=Path("data/releases/reactionfusion_neural_combined_v5"),
    )
    args = parser.parse_args()
    report = train_and_release_combined(args.dataset, args.config, args.release_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
