"""Train, cross-validate, serialize, and release ReactionFusion v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from reactionfusion.evaluation.human_validation import (
    SENTIMENT_LABELS,
    _load_and_validate_annotations,
    classification_metrics,
)
from reactionfusion.labeling.reactionfusion_v2 import (
    REACTIONS,
    ReactionFusionV2Model,
    feature_matrix,
    predict_v2_matrix,
    train_v2_model,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_group_folds(
    labels: pd.Series, groups: pd.Series, *, n_splits: int, seed: int
) -> np.ndarray:
    """Assign complete text groups to deterministic approximately stratified folds."""
    frame = pd.DataFrame(
        {
            "label": labels.astype("string").str.strip().str.lower(),
            "group": groups.astype(str),
        }
    )
    group_rows = frame.groupby("group", as_index=False).agg(label=("label", "first"))
    assignments: dict[str, int] = {}
    for _, stratum in group_rows.groupby("label", sort=True):
        ordered = sorted(stratum["group"].tolist(), key=lambda value: _stable_key(value, seed))
        for index, group in enumerate(ordered):
            assignments[group] = index % n_splits
    folds = frame["group"].map(assignments).to_numpy(dtype=int)
    if len(set(folds)) != n_splits:
        raise ValueError("Unable to create every requested cross-validation fold")
    return folds


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / max(temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def select_temperature(
    probabilities: np.ndarray,
    truth: pd.Series,
    classes: tuple[str, ...],
    candidates: list[float],
) -> tuple[float, dict[str, float]]:
    class_to_id = {label: index for index, label in enumerate(classes)}
    normalized = truth.astype("string").str.strip().str.lower()
    usable = normalized.isin(classes).to_numpy()
    target_ids = np.asarray([class_to_id[label] for label in normalized[usable]], dtype=int)
    losses: dict[str, float] = {}
    for temperature in candidates:
        calibrated = _apply_temperature(probabilities[usable], temperature)
        selected = calibrated[np.arange(len(target_ids)), target_ids]
        loss = -float(np.log(np.clip(selected, 1e-12, 1.0)).mean())
        losses[str(temperature)] = loss
    best = min(candidates, key=lambda value: losses[str(value)])
    return float(best), losses


def select_abstention_threshold(
    probabilities: np.ndarray, config: Mapping[str, Any]
) -> float:
    target_coverage = float(config["target_coverage"])
    threshold = float(np.quantile(probabilities.max(axis=1), 1.0 - target_coverage))
    return max(
        float(config["minimum_abstention_threshold"]),
        min(float(config["maximum_abstention_threshold"]), threshold),
    )


def _binary_metrics(truth: pd.Series, probabilities: np.ndarray) -> dict[str, float | int]:
    normalized = truth.astype("string").str.strip().str.lower()
    usable = normalized.isin({"yes", "no"}).to_numpy()
    target = (normalized[usable] == "yes").to_numpy()
    prediction = probabilities[usable] >= 0.5
    true_positive = int((target & prediction).sum())
    false_positive = int((~target & prediction).sum())
    false_negative = int((target & ~prediction).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "evaluated_records": int(usable.sum()),
        "positive_support": int(target.sum()),
        "accuracy": float((target == prediction).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _paired_accuracy_test(
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
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(first_only, second_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "discordant_records": discordant,
        "two_sided_exact_mcnemar_p_value": p_value,
    }


def _cross_validate(
    annotated: pd.DataFrame,
    base_features: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, float, float]:
    folds = assign_group_folds(
        annotated["sentiment"],
        annotated["text_hash"],
        n_splits=int(config["cross_validation_folds"]),
        seed=int(config["seed"]),
    )
    classes = tuple(config["sentiment_classes"])
    emotion_targets = tuple(config["emotion_targets"])
    probabilities = np.full((len(annotated), len(classes)), np.nan)
    emotion_probabilities = np.full((len(annotated), len(emotion_targets)), np.nan)
    fold_summaries = []

    for fold in range(int(config["cross_validation_folds"])):
        validation = folds == fold
        training = ~validation
        train_groups = set(annotated.loc[training, "text_hash"])
        validation_groups = set(annotated.loc[validation, "text_hash"])
        if train_groups & validation_groups:
            raise AssertionError("Text group leaked across cross-validation folds")
        model = train_v2_model(
            base_features[training],
            annotated.loc[training].reset_index(drop=True),
            config,
            metadata={"cross_validation_fold": fold},
        )
        fold_probabilities, fold_emotions = predict_v2_matrix(model, base_features[validation])
        probabilities[validation] = fold_probabilities
        emotion_probabilities[validation] = fold_emotions
        fold_summaries.append(
            {
                "fold": fold,
                "training_records": int(training.sum()),
                "validation_records": int(validation.sum()),
                "training_groups": len(train_groups),
                "validation_groups": len(validation_groups),
            }
        )

    if np.isnan(probabilities).any() or np.isnan(emotion_probabilities).any():
        raise AssertionError("Cross-validation did not produce every out-of-fold prediction")
    temperature, temperature_losses = select_temperature(
        probabilities,
        annotated["sentiment"],
        classes,
        [float(value) for value in config["temperature_candidates"]],
    )
    calibrated = _apply_temperature(probabilities, temperature)
    threshold = select_abstention_threshold(calibrated, config)
    candidate_ids = calibrated.argmax(axis=1)
    candidates = np.asarray([classes[index] for index in candidate_ids])
    confidence = calibrated.max(axis=1)
    abstained = confidence < threshold
    final_labels = np.where(abstained, "uncertain", candidates)

    oof = annotated.copy()
    oof["cv_fold"] = folds
    oof["v2_candidate_sentiment"] = candidates
    oof["v2_sentiment_label"] = final_labels
    oof["v2_confidence"] = confidence
    oof["v2_is_uncertain"] = abstained
    for index, label in enumerate(classes):
        oof[f"v2_probability_{label}"] = calibrated[:, index]
    for index, target in enumerate(emotion_targets):
        oof[f"v2_emotion_probability_{target}"] = emotion_probabilities[:, index]

    normalized_truth = annotated["sentiment"].astype("string").str.strip().str.lower()
    evaluable = normalized_truth.isin(SENTIMENT_LABELS).to_numpy()
    covered = evaluable & ~abstained
    emotion_metrics = {
        target: _binary_metrics(annotated[target], emotion_probabilities[:, index])
        for index, target in enumerate(emotion_targets)
    }
    report = {
        "evaluation_design": (
            "five_fold_grouped_out_of_fold_on_uncertainty_enriched_development_set"
        ),
        "folds": fold_summaries,
        "group_leakage_count": 0,
        "temperature": temperature,
        "temperature_log_losses": temperature_losses,
        "abstention_threshold": threshold,
        "target_coverage": float(config["target_coverage"]),
        "observed_coverage": float(covered.sum() / evaluable.sum()),
        "covered_records": int(covered.sum()),
        "selective_accuracy": float(
            (final_labels[covered] == normalized_truth.to_numpy()[covered]).mean()
        ),
        "candidate_sentiment_metrics": classification_metrics(
            annotated["sentiment"], pd.Series(candidates)
        ),
        "abstaining_sentiment_metrics": classification_metrics(
            annotated["sentiment"], pd.Series(final_labels)
        ),
        "reactionfusion_v1_metrics": classification_metrics(
            annotated["sentiment"], annotated["sentiment_label"]
        ),
        "filtered_baseline_metrics": classification_metrics(
            annotated["sentiment"], annotated["baseline_filtered_label"]
        ),
        "paired_accuracy_v2_vs_v1": _paired_accuracy_test(
            annotated["sentiment"], pd.Series(candidates), annotated["sentiment_label"]
        ),
        "paired_accuracy_v2_vs_filtered_baseline": _paired_accuracy_test(
            annotated["sentiment"],
            pd.Series(candidates),
            annotated["baseline_filtered_label"],
        ),
        "emotion_metrics": emotion_metrics,
        "candidate_distribution": dict(Counter(candidates)),
        "final_distribution": dict(Counter(final_labels)),
    }
    return report, oof, temperature, threshold


def _apply_final_model(
    dataset: pd.DataFrame, model: ReactionFusionV2Model
) -> pd.DataFrame:
    base_features = feature_matrix(dataset, model.smoothing_alpha)
    sentiment_probabilities, emotion_probabilities = predict_v2_matrix(model, base_features)
    classes = model.sentiment_model.classes
    candidate_ids = sentiment_probabilities.argmax(axis=1)
    candidates = np.asarray([classes[index] for index in candidate_ids])
    confidence = sentiment_probabilities.max(axis=1)
    uncertain = confidence < model.abstention_threshold
    labels = np.where(uncertain, "uncertain", candidates)

    rename = {
        "sentiment_label": "sentiment_label_v1",
        "label_confidence": "label_confidence_v1",
        "fusion_score": "fusion_score_v1",
        "reaction_entropy": "reaction_entropy_v1",
        "is_ambiguous": "is_ambiguous_v1",
        "positive_anchor_mass": "positive_anchor_mass_v1",
        "negative_anchor_mass": "negative_anchor_mass_v1",
        "label_version": "label_version_v1",
    }
    output = dataset.rename(columns=rename).copy()
    output["sentiment_label"] = labels
    output["candidate_sentiment_label"] = candidates
    output["label_confidence"] = confidence
    output["is_uncertain"] = uncertain
    output["is_ambiguous"] = uncertain | (candidates == "mixed")
    entropy_index = model.base_feature_names.index("reaction_entropy")
    output["reaction_entropy"] = base_features[:, entropy_index]
    output["mixed_evidence"] = base_features[:, model.base_feature_names.index("mixed_evidence")]
    for index, label in enumerate(classes):
        output[f"sentiment_probability_{label}"] = sentiment_probabilities[:, index]
    for index, target in enumerate(model.emotion_targets):
        output[f"emotion_probability_{target}"] = emotion_probabilities[:, index]
    reasons = []
    anchor_index = model.base_feature_names.index("anchor_balance")
    mixed_index = model.base_feature_names.index("mixed_evidence")
    for row_index, candidate in enumerate(candidates):
        ranked = np.argsort(emotion_probabilities[row_index])[::-1][:2]
        strongest = ", ".join(
            f"{model.emotion_targets[index]}={emotion_probabilities[row_index, index]:.2f}"
            for index in ranked
        )
        reasons.append(
            f"candidate={candidate}; anchor_balance={base_features[row_index, anchor_index]:.3f}; "
            f"mixed_evidence={base_features[row_index, mixed_index]:.3f}; strongest={strongest}"
        )
    output["decision_reason"] = reasons
    output["prediction_source"] = "final_model"
    output["label_version"] = model.version
    return output


def _apply_oof_overrides(
    release: pd.DataFrame, oof: pd.DataFrame, model: ReactionFusionV2Model
) -> pd.DataFrame:
    """Use cross-fitted predictions for every record used to calibrate v2."""
    output = release.copy()
    lookup = oof.set_index("record_id")
    matched = output["record_id"].isin(lookup.index)
    record_ids = output.loc[matched, "record_id"]
    output.loc[matched, "candidate_sentiment_label"] = record_ids.map(
        lookup["v2_candidate_sentiment"]
    ).to_numpy()
    output.loc[matched, "sentiment_label"] = record_ids.map(
        lookup["v2_sentiment_label"]
    ).to_numpy()
    output.loc[matched, "label_confidence"] = record_ids.map(
        lookup["v2_confidence"]
    ).to_numpy()
    output.loc[matched, "is_uncertain"] = record_ids.map(
        lookup["v2_is_uncertain"]
    ).to_numpy(dtype=bool)
    for label in model.sentiment_model.classes:
        output.loc[matched, f"sentiment_probability_{label}"] = record_ids.map(
            lookup[f"v2_probability_{label}"]
        ).to_numpy()
    for target in model.emotion_targets:
        output.loc[matched, f"emotion_probability_{target}"] = record_ids.map(
            lookup[f"v2_emotion_probability_{target}"]
        ).to_numpy()
    output.loc[matched, "is_ambiguous"] = (
        output.loc[matched, "is_uncertain"].astype(bool)
        | (output.loc[matched, "candidate_sentiment_label"] == "mixed")
    )
    output.loc[matched, "prediction_source"] = "grouped_out_of_fold"
    for row_index in output.index[matched]:
        ranked = sorted(
            (
                (target, float(output.at[row_index, f"emotion_probability_{target}"]))
                for target in model.emotion_targets
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        strongest = ", ".join(f"{name}={value:.2f}" for name, value in ranked)
        output.at[row_index, "decision_reason"] = (
            f"grouped_out_of_fold; candidate={output.at[row_index, 'candidate_sentiment_label']}; "
            f"mixed_evidence={output.at[row_index, 'mixed_evidence']:.3f}; strongest={strongest}"
        )
    return output


def _dataset_card(report: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    metrics = report["candidate_sentiment_metrics"]
    abstaining = report["abstaining_sentiment_metrics"]
    v1 = report["reactionfusion_v1_metrics"]
    baseline = report["filtered_baseline_metrics"]
    v2_v1_p = report["paired_accuracy_v2_vs_v1"]["two_sided_exact_mcnemar_p_value"]
    v2_baseline_p = report["paired_accuracy_v2_vs_filtered_baseline"][
        "two_sided_exact_mcnemar_p_value"
    ]
    return f"""# ReactionFusion v2 development dataset card

## Status

This is an **experimental development release**, not a final benchmark dataset.
ReactionFusion v2 is a reaction-only hybrid model calibrated with 120
uncertainty-enriched human annotations. It does not use post text as an input.

## Architecture

- Laplace-smoothed ratios for all seven reactions.
- Clear-valence anchors and context gates for Like, Haha, and Wow.
- Entropy, opposition, dominance, engagement, and interaction features.
- Eleven regularized human-calibrated emotion/stance probability models.
- Four-class regularized sentiment fusion: negative, neutral, positive, mixed.
- Temperature calibration and confidence-based abstention.
- Cross-fitted labels for all human-calibration records to prevent in-sample labels.

## Cross-validated development results

- Evaluation: grouped out-of-fold predictions on the annotated difficult-case set.
- Abstaining-label accuracy: {abstaining['accuracy']:.3f}
- Abstaining-label macro F1: {abstaining['macro_f1_four_classes']:.3f}
- Observed confident coverage: {report['observed_coverage']:.3f}
- Accuracy on covered records: {report['selective_accuracy']:.3f}
- Group leakage: {report['group_leakage_count']}

| Method | Accuracy | Four-class macro F1 |
|---|---:|---:|
| ReactionFusion v2 candidate | {metrics['accuracy']:.3f} | {metrics['macro_f1_four_classes']:.3f} |
| ReactionFusion v1 | {v1['accuracy']:.3f} | {v1['macro_f1_four_classes']:.3f} |
| Filtered baseline | {baseline['accuracy']:.3f} | {baseline['macro_f1_four_classes']:.3f} |

Paired exact McNemar p-values are {v2_v1_p:.3f} for v2 versus v1 and
{v2_baseline_p:.3f} for v2 versus the filtered baseline. These development-set
statistics do not replace evaluation on a representative frozen human test set.

## Generated label distribution

{json.dumps(quality['label_distribution'], ensure_ascii=False)}

## Limitations

- The calibration sample was selected for uncertainty and is not representative.
- Only nine adjudicated mixed examples are available.
- Surprise, fear, and care/empathy have very few positive human examples; their
  probabilities are exploratory rather than validated emotion classifiers.
- V2 must be evaluated once on a separately frozen representative human test set.
- Records labeled `uncertain` should be excluded from downstream ANN training.
- Source licensing, privacy, platform terms, and ethics requirements still apply.

## Files

- `dataset.csv`: complete v2 development release with v1 audit columns.
- `dataset_deduplicated.csv`: one record per normalized text.
- `train.csv`, `validation.csv`, `test.csv`: existing leakage-safe splits.
- `model.json`: serialized v2 feature, emotion, sentiment, and calibration model.
- `training_config.json`: frozen training hyperparameters.
- `cross_validation_results.json`: complete grouped out-of-fold metrics.
- `human_validation_oof.csv`: record-level out-of-fold predictions.
- `quality_report.json`: generated release counts and distributions.
"""


def train_and_release_v2(
    dataset_path: Path,
    annotation_path: Path,
    config_path: Path,
    release_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = pd.read_csv(dataset_path)
    required_counts = [f"{reaction}_count" for reaction in REACTIONS]
    required = ("record_id", "text_hash", "split", *required_counts)
    missing = [column for column in required if column not in dataset]
    if missing:
        raise ValueError(f"V1 dataset is missing required v2 inputs: {missing}")
    annotations = _load_and_validate_annotations(annotation_path)["Adjudication"]
    annotated = annotations.merge(
        dataset,
        on="record_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_dataset"),
    )
    if annotated[required_counts].isna().any().any():
        raise ValueError("Annotated records are missing reaction counts")

    base_features = feature_matrix(annotated, float(config["smoothing_alpha"]))
    cv_report, oof, temperature, threshold = _cross_validate(annotated, base_features, config)
    model = train_v2_model(
        base_features,
        annotated,
        config,
        metadata={
            "annotation_workbook_sha256": _sha256(annotation_path),
            "v1_dataset_sha256": _sha256(dataset_path),
            "training_records": len(annotated),
            "sample_design": "uncertainty_enriched_development",
        },
    )
    model = replace(model, temperature=temperature, abstention_threshold=threshold)
    release = _apply_final_model(dataset, model)
    release = _apply_oof_overrides(release, oof, model)
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
        json.dumps(cv_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    quality = {
        "records": len(release),
        "deduplicated_records": len(deduplicated),
        "label_distribution": dict(Counter(release["sentiment_label"])),
        "candidate_distribution": dict(Counter(release["candidate_sentiment_label"])),
        "uncertain_records": int(release["is_uncertain"].sum()),
        "confident_records": int((~release["is_uncertain"]).sum()),
        "split_distribution": release["split"].value_counts().sort_index().to_dict(),
        "split_label_distribution": pd.crosstab(
            release["split"], release["sentiment_label"]
        ).to_dict(orient="index"),
        "prediction_source_distribution": dict(Counter(release["prediction_source"])),
        "version": model.version,
        "source_v1_dataset_sha256": _sha256(dataset_path),
        "annotation_workbook_sha256": _sha256(annotation_path),
    }
    (release_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (release_dir / "DATASET_CARD.md").write_text(
        _dataset_card(cv_report, quality), encoding="utf-8"
    )
    return {"cross_validation": cv_report, "quality": quality}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/releases/reactionfusion_v1/dataset.csv"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/reactionfusion_v1/adjudication_completed.xlsx"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/labeling/reactionfusion_v2.json"),
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=Path("data/releases/reactionfusion_v2"),
    )
    args = parser.parse_args()
    report = train_and_release_v2(
        args.dataset, args.annotations, args.config, args.release_dir
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
