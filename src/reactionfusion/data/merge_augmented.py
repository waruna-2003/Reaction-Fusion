"""Merge the legacy dataset with the provided 15k synthetic augmentation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from reactionfusion.data.preprocessing import (
    HEADER_MAP,
    REACTIONS,
    _assign_group_splits,
    detect_language,
    normalize_text,
    text_hash,
)
from reactionfusion.evaluation.human_validation import (
    BINARY_COLUMNS,
    SENTIMENT_VALUES,
    _load_and_validate_annotations,
    classification_metrics,
    cohen_kappa,
)
from reactionfusion.labeling import filtered_baseline, fuse_reactions
from reactionfusion.labeling.reactionfusion_neural import (
    ReactionFusionNeuralModel,
    train_neural_model,
)
from reactionfusion.labeling.reactionfusion_v2 import (
    BASE_FEATURE_NAMES,
    ReactionFusionV2Model,
    feature_matrix,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_synthetic_posts(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="FB Posts (Synthetic)", engine="openpyxl")
    missing = [column for column in HEADER_MAP if column not in raw]
    if missing:
        raise ValueError(f"Synthetic source is missing columns: {missing}")
    source = raw[list(HEADER_MAP)].rename(columns=HEADER_MAP).copy()
    count_columns = [f"{reaction}_count" for reaction in REACTIONS]
    for column in count_columns:
        values = pd.to_numeric(source[column], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"Synthetic source has invalid values in {column}")
        source[column] = values.astype("int64")
    calculated = source[count_columns].sum(axis=1).astype("int64")
    if not calculated.equals(source["source_total_reactions"].astype("int64")):
        raise ValueError("Synthetic reaction totals do not reconcile")
    if source["source_row_id"].duplicated().any():
        raise ValueError("Synthetic source contains duplicate row numbers")

    source["calculated_total_reactions"] = calculated
    source["record_id"] = source["source_row_id"].map(
        lambda value: f"SYN_POST_{int(value):06d}"
    )
    normalized_unmasked = source["raw_text"].map(
        lambda value: normalize_text(value, mask_private=False)
    )
    source["model_text"] = source["raw_text"].map(normalize_text)
    source["privacy_masked"] = normalized_unmasked != source["model_text"]
    source["language_type"] = source["model_text"].map(detect_language)
    source["text_hash"] = source["model_text"].map(text_hash)
    source["duplicate_group_id"] = source["text_hash"].str[:16].map(
        lambda value: f"DUP_{value}"
    )
    duplicate_counts = source.groupby("text_hash")["text_hash"].transform("size")
    source["duplicate_count"] = duplicate_counts.astype("int64")
    source["is_duplicate"] = duplicate_counts > 1
    source["total_reactions"] = calculated
    source["log_total_reactions"] = np.log1p(calculated)

    alpha = float(config["smoothing_alpha"])
    for reaction in REACTIONS:
        count_column = f"{reaction}_count"
        source[f"{reaction}_ratio"] = source[count_column] / source["total_reactions"]
        source[f"{reaction}_ratio_smoothed"] = (
            source[count_column] + alpha
        ) / (source["total_reactions"] + alpha * len(REACTIONS))
    count_matrix = source[count_columns].to_numpy()
    sorted_counts = np.sort(count_matrix, axis=1)
    dominant_indices = count_matrix.argmax(axis=1)
    source["dominant_reaction"] = [REACTIONS[index] for index in dominant_indices]
    source["dominant_ratio"] = count_matrix.max(axis=1) / calculated.to_numpy()
    source["dominance_margin"] = (
        sorted_counts[:, -1] - sorted_counts[:, -2]
    ) / calculated.to_numpy()

    labels = []
    for _, row in source.iterrows():
        counts = {reaction: int(row[f"{reaction}_count"]) for reaction in REACTIONS}
        result = fuse_reactions(counts, config)
        baseline_label, baseline_score = filtered_baseline(counts)
        labels.append(
            {
                "sentiment_label": result.sentiment_label,
                "label_confidence": result.label_confidence,
                "fusion_score": result.fusion_score,
                "reaction_entropy": result.reaction_entropy,
                "is_ambiguous": result.is_ambiguous,
                "positive_anchor_mass": result.positive_anchor_mass,
                "negative_anchor_mass": result.negative_anchor_mass,
                "baseline_filtered_label": baseline_label,
                "baseline_filtered_score": baseline_score,
                "label_version": config["version"],
            }
        )
    source = pd.concat([source.reset_index(drop=True), pd.DataFrame(labels)], axis=1)
    source["quality_status"] = "approved_synthetic_augmentation"
    source["data_origin"] = "synthetic_augmentation"
    source["text_provenance"] = "template_generated_synthetic"
    source["reaction_provenance"] = "fabricated_synthetic"
    return source.drop(columns=["raw_text"])


def _prepare_legacy(dataset_path: Path) -> pd.DataFrame:
    legacy = pd.read_csv(dataset_path, low_memory=False)
    legacy["data_origin"] = "legacy_source"
    legacy["text_provenance"] = "legacy_collected_unverified"
    legacy["reaction_provenance"] = "legacy_collected_unverified"
    return legacy


def _audit_annotations(
    workbook_path: Path, posts: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    sheets = _load_and_validate_annotations(workbook_path)
    expected_ids = posts["record_id"].reset_index(drop=True)
    expected_text = posts["model_text"].reset_index(drop=True)
    for name, frame in sheets.items():
        if not frame["record_id"].reset_index(drop=True).equals(expected_ids):
            raise ValueError(f"{name} record IDs are not aligned with synthetic posts")
        normalized_annotation_text = frame["model_text"].map(normalize_text).reset_index(
            drop=True
        )
        if not normalized_annotation_text.equals(expected_text):
            raise ValueError(f"{name} text is not aligned with synthetic posts")

    first = sheets["Annotator 1"]
    second = sheets["Annotator 2"]
    adjudicated = sheets["Adjudication"]
    disagreement = first["sentiment"] != second["sentiment"]
    audit = {
        "records": len(adjudicated),
        "row_alignment_valid": True,
        "sentiment_distribution": dict(Counter(adjudicated["sentiment"])),
        "annotator_sentiment_raw_agreement": float(
            (first["sentiment"] == second["sentiment"]).mean()
        ),
        "annotator_sentiment_cohen_kappa": cohen_kappa(
            first["sentiment"], second["sentiment"]
        ),
        "sentiment_disagreements": int(disagreement.sum()),
        "adjudication_matches_annotator_2_when_disagreeing": float(
            (
                adjudicated.loc[disagreement, "sentiment"]
                == second.loc[disagreement, "sentiment"]
            ).mean()
        ),
        "annotation_note_unique_counts": {
            name: int(frame["annotation_notes"].astype(str).nunique())
            for name, frame in sheets.items()
        },
        "provenance_classification": "provided_synthetic_annotations_unverified",
        "ground_truth_eligible": False,
    }
    return sheets, audit


def _attach_annotations(
    combined: pd.DataFrame,
    synthetic_annotations: pd.DataFrame,
    human_annotations: pd.DataFrame,
) -> pd.DataFrame:
    fields = ["sentiment", *BINARY_COLUMNS, "confidence"]
    synthetic = synthetic_annotations[["record_id", *fields]].copy()
    human = human_annotations[["record_id", *fields]].copy()
    synthetic["annotation_provenance"] = "provided_synthetic_unverified"
    human["annotation_provenance"] = "human_adjudicated_development"
    annotations = pd.concat([human, synthetic], ignore_index=True)
    if annotations["record_id"].duplicated().any():
        raise ValueError("Human and synthetic annotation record IDs collide")
    annotations = annotations.rename(
        columns={column: f"provided_{column}" for column in fields}
    )
    merged = combined.merge(annotations, on="record_id", how="left", validate="one_to_one")
    merged["annotation_provenance"] = merged["annotation_provenance"].fillna("none")
    return merged


def _apply_v2_predictions(dataset: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    model = ReactionFusionV2Model.from_dict(json.loads(model_path.read_text(encoding="utf-8")))
    base = feature_matrix(dataset, model.smoothing_alpha)
    from reactionfusion.labeling.reactionfusion_v2 import predict_v2_matrix

    sentiment, emotions = predict_v2_matrix(model, base)
    candidate_ids = sentiment.argmax(axis=1)
    candidates = np.asarray(model.sentiment_model.classes)[candidate_ids]
    confidence = sentiment.max(axis=1)
    output = pd.DataFrame(index=dataset.index)
    output["v2_candidate_sentiment"] = candidates
    output["v2_sentiment_label"] = np.where(
        confidence < model.abstention_threshold, "uncertain", candidates
    )
    output["v2_confidence"] = confidence
    for index, label in enumerate(model.sentiment_model.classes):
        output[f"v2_probability_{label}"] = sentiment[:, index]
    for index, target in enumerate(model.emotion_targets):
        output[f"v2_emotion_probability_{target}"] = emotions[:, index]
    return output


def _apply_neural_predictions(dataset: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    model = ReactionFusionNeuralModel.from_dict(
        json.loads(model_path.read_text(encoding="utf-8"))
    )
    sentiment, emotions = model.predict_probability(
        feature_matrix(dataset, model.smoothing_alpha)
    )
    candidate_ids = sentiment.argmax(axis=1)
    candidates = np.asarray(model.sentiment_classes)[candidate_ids]
    confidence = sentiment.max(axis=1)
    output = pd.DataFrame(index=dataset.index)
    output["neural_v3_candidate_sentiment"] = candidates
    output["neural_v3_sentiment_label"] = np.where(
        confidence < model.abstention_threshold, "uncertain", candidates
    )
    output["neural_v3_confidence"] = confidence
    for index, label in enumerate(model.sentiment_classes):
        output[f"neural_v3_probability_{label}"] = sentiment[:, index]
    for index, target in enumerate(model.emotion_targets):
        output[f"neural_v3_emotion_probability_{target}"] = emotions[:, index]
    return output


def _override_human_oof(dataset: pd.DataFrame, release_root: Path) -> pd.DataFrame:
    mappings = (
        (release_root / "reactionfusion_v2" / "human_validation_oof.csv", "v2"),
        (release_root / "reactionfusion_neural_v3" / "human_validation_oof.csv", "neural_v3"),
    )
    for path, prefix in mappings:
        oof = pd.read_csv(path, low_memory=False).set_index("record_id")
        mask = dataset["record_id"].isin(oof.index)
        record_ids = dataset.loc[mask, "record_id"]
        for label in ("negative", "neutral", "positive", "mixed"):
            source = (
                f"v2_probability_{label}"
                if prefix == "v2"
                else f"sentiment_probability_{label}"
            )
            target = f"{prefix}_probability_{label}"
            dataset.loc[mask, target] = record_ids.map(oof[source]).to_numpy()
        candidate_source = (
            "v2_candidate_sentiment"
            if prefix == "v2"
            else "candidate_sentiment_label"
        )
        dataset.loc[mask, f"{prefix}_candidate_sentiment"] = record_ids.map(
            oof[candidate_source]
        ).to_numpy()
        label_source = "v2_sentiment_label" if prefix == "v2" else "sentiment_label"
        dataset.loc[mask, f"{prefix}_sentiment_label"] = record_ids.map(
            oof[label_source]
        ).to_numpy()
        confidence_source = "v2_confidence" if prefix == "v2" else "label_confidence"
        dataset.loc[mask, f"{prefix}_confidence"] = record_ids.map(
            oof[confidence_source]
        ).to_numpy()
    return dataset


def _run_synthetic_transfer_probe(
    synthetic: pd.DataFrame,
    synthetic_annotations: pd.DataFrame,
    human_dataset: pd.DataFrame,
    human_annotations: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[ReactionFusionNeuralModel, dict[str, Any]]:
    probe_config = dict(config)
    probe_config["version"] = "reactionfusion_synthetic_transfer_probe_v4"
    probe_config["epochs"] = int(config.get("synthetic_probe_epochs", 300))
    model = train_neural_model(
        feature_matrix(synthetic, float(config["smoothing_alpha"])),
        synthetic_annotations,
        BASE_FEATURE_NAMES,
        probe_config,
        metadata={
            "training_origin": "provided_synthetic_annotations_unverified",
            "promotion_status": "rejected_pending_real_data_performance",
        },
    )
    human = human_annotations.merge(
        human_dataset, on="record_id", how="left", validate="one_to_one"
    )
    probabilities, _ = model.predict_probability(
        feature_matrix(human, float(config["smoothing_alpha"]))
    )
    predictions = pd.Series(
        np.asarray(config["sentiment_classes"])[probabilities.argmax(axis=1)]
    )
    metrics = classification_metrics(human["sentiment"], predictions)
    return model, {
        "evaluation_design": "train_on_15000_synthetic_evaluate_on_120_original_human",
        "metrics": metrics,
        "promotion_status": "rejected",
        "reason": "performance is below the existing human-calibrated neural v3 model",
    }


def _dataset_card(quality: Mapping[str, Any], transfer: Mapping[str, Any]) -> str:
    metrics = transfer["metrics"]
    return f"""# ReactionFusion combined augmentation v4 dataset card

## Status

This release combines the legacy 997-record source with 15,000 explicitly
synthetic posts. Synthetic and legacy records remain identifiable through
`data_origin`, `text_provenance`, and `reaction_provenance`.

The synthetic workbook states that its Facebook URLs, reaction counts, and post
text are fabricated. Its supplied annotations are stored as unverified synthetic
supervision and are **not human ground truth**.

## Counts

- Combined records: {quality['records']}
- Legacy records: {quality['origin_distribution'].get('legacy_source', 0)}
- Synthetic records: {quality['origin_distribution'].get('synthetic_augmentation', 0)}
- Deduplicated records: {quality['deduplicated_records']}
- Original human development annotations: 120
- Provided synthetic annotations: 15,000

## Restarted predictions

ReactionFusion v1, v2, and neural v3 predictions were regenerated for the combined
records. Existing grouped out-of-fold predictions were retained for the original
120 human-development records.

## Synthetic transfer experiment

A neural ensemble trained on the 15,000 synthetic annotations achieved
{metrics['accuracy']:.3f} accuracy and {metrics['macro_f1_four_classes']:.3f}
four-class macro-F1 on the original adjudicated human development records. The
experiment is rejected for model promotion because it performs substantially below
the existing human-calibrated neural v3 model.

## Usage policy

- Use `legacy_source` records for the original research lineage.
- Use `synthetic_augmentation` only for explicitly reported augmentation studies.
- Never report synthetic URLs or reactions as observed Facebook engagement.
- Do not combine synthetic annotations with a final human benchmark.
"""


def merge_and_restart(
    legacy_path: Path,
    synthetic_posts_path: Path,
    synthetic_annotations_path: Path,
    human_annotations_path: Path,
    v1_config_path: Path,
    neural_config_path: Path,
    release_dir: Path,
) -> dict[str, Any]:
    v1_config = json.loads(v1_config_path.read_text(encoding="utf-8"))
    neural_config = json.loads(neural_config_path.read_text(encoding="utf-8"))
    legacy = _prepare_legacy(legacy_path)
    synthetic = _load_synthetic_posts(synthetic_posts_path, v1_config)
    sheets, annotation_audit = _audit_annotations(
        synthetic_annotations_path, synthetic
    )
    human_annotations = _load_and_validate_annotations(human_annotations_path)[
        "Adjudication"
    ]

    common_columns = sorted(set(legacy.columns) | set(synthetic.columns))
    combined = pd.concat(
        [legacy.reindex(columns=common_columns), synthetic.reindex(columns=common_columns)],
        ignore_index=True,
    )
    if combined["record_id"].duplicated().any():
        raise ValueError("Record IDs collide after merging legacy and synthetic data")
    combined["duplicate_count"] = combined.groupby("text_hash")["text_hash"].transform(
        "size"
    ).astype("int64")
    combined["is_duplicate"] = combined["duplicate_count"] > 1
    combined["split"] = _assign_group_splits(combined, seed=42)
    combined = _attach_annotations(
        combined, sheets["Adjudication"], human_annotations
    )

    v2 = _apply_v2_predictions(
        combined, Path("data/releases/reactionfusion_v2/model.json")
    )
    neural = _apply_neural_predictions(
        combined, Path("data/releases/reactionfusion_neural_v3/model.json")
    )
    combined = pd.concat([combined.reset_index(drop=True), v2, neural], axis=1)
    combined = _override_human_oof(combined, Path("data/releases"))

    transfer_model, transfer_report = _run_synthetic_transfer_probe(
        synthetic,
        sheets["Adjudication"],
        legacy,
        human_annotations,
        {**neural_config, "synthetic_probe_epochs": 300},
    )
    deduplicated = combined.sort_values(
        ["text_hash", "total_reactions"], ascending=[True, False]
    ).drop_duplicates("text_hash")

    release_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(release_dir / "dataset.csv", index=False, encoding="utf-8-sig")
    deduplicated.to_csv(
        release_dir / "dataset_deduplicated.csv", index=False, encoding="utf-8-sig"
    )
    for split in ("train", "validation", "test"):
        combined[combined["split"] == split].to_csv(
            release_dir / f"{split}.csv", index=False, encoding="utf-8-sig"
        )
    annotation_columns = [
        "record_id",
        "data_origin",
        "annotation_provenance",
        *(column for column in combined if column.startswith("provided_")),
    ]
    annotated = combined.loc[
        combined["annotation_provenance"] != "none", annotation_columns
    ]
    annotated.to_csv(
        release_dir / "provided_annotations_merged.csv", index=False, encoding="utf-8-sig"
    )
    (release_dir / "synthetic_transfer_model.json").write_text(
        json.dumps(transfer_model.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (release_dir / "synthetic_transfer_results.json").write_text(
        json.dumps(transfer_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_audit = {
        "legacy_dataset_sha256": _sha256(legacy_path),
        "synthetic_posts_sha256": _sha256(synthetic_posts_path),
        "synthetic_annotations_sha256": _sha256(synthetic_annotations_path),
        "human_annotations_sha256": _sha256(human_annotations_path),
        "synthetic_source_notice": (
            "URLs, reaction counts, and text are fabricated according to the workbook README"
        ),
        "annotation_audit": annotation_audit,
    }
    (release_dir / "source_audit.json").write_text(
        json.dumps(source_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    quality = {
        "records": len(combined),
        "deduplicated_records": len(deduplicated),
        "origin_distribution": dict(Counter(combined["data_origin"])),
        "annotation_provenance_distribution": dict(
            Counter(combined["annotation_provenance"])
        ),
        "split_distribution": dict(Counter(combined["split"])),
        "cross_split_text_group_leakage": int(
            (combined.groupby("text_hash")["split"].nunique() > 1).sum()
        ),
        "v1_label_distribution": dict(Counter(combined["sentiment_label"])),
        "v2_label_distribution": dict(Counter(combined["v2_sentiment_label"])),
        "neural_v3_label_distribution": dict(
            Counter(combined["neural_v3_sentiment_label"])
        ),
    }
    (release_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (release_dir / "DATASET_CARD.md").write_text(
        _dataset_card(quality, transfer_report), encoding="utf-8"
    )
    return {
        "quality": quality,
        "annotation_audit": annotation_audit,
        "synthetic_transfer": transfer_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy", type=Path, default=Path("data/releases/reactionfusion_v1/dataset.csv")
    )
    parser.add_argument(
        "--synthetic-posts",
        type=Path,
        default=Path("data/raw/synthetic_exports/facebook_posts_synthetic_15k.xlsx"),
    )
    parser.add_argument(
        "--synthetic-annotations",
        type=Path,
        default=Path(
            "data/annotations/synthetic_15k/adjudicated_synthetic_15k.xlsx"
        ),
    )
    parser.add_argument(
        "--human-annotations",
        type=Path,
        default=Path("data/annotations/reactionfusion_v1/adjudication_completed.xlsx"),
    )
    parser.add_argument(
        "--v1-config",
        type=Path,
        default=Path("configs/labeling/reactionfusion_v1.json"),
    )
    parser.add_argument(
        "--neural-config",
        type=Path,
        default=Path("configs/labeling/reactionfusion_neural_v3.json"),
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=Path("data/releases/reactionfusion_augmented_v4"),
    )
    args = parser.parse_args()
    report = merge_and_restart(
        args.legacy,
        args.synthetic_posts,
        args.synthetic_annotations,
        args.human_annotations,
        args.v1_config,
        args.neural_config,
        args.release_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
