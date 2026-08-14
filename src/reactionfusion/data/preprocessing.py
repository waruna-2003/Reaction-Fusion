"""End-to-end preprocessing for the initial ReactionFusion Facebook-post dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reactionfusion.labeling import filtered_baseline, fuse_reactions


REACTIONS = ("like", "love", "care", "haha", "wow", "sad", "angry")
REACTION_COLUMNS = tuple(f"{reaction}_count" for reaction in REACTIONS)
HEADER_MAP = {
    "#": "source_row_id",
    "Post Text": "raw_text",
    "Likes": "like_count",
    "Love": "love_count",
    "Care": "care_count",
    "Haha": "haha_count",
    "Wow": "wow_count",
    "Sad": "sad_count",
    "Angry": "angry_count",
    "Total Reactions": "source_total_reactions",
}
PHONE_RE = re.compile(r"(?:\+?94|0)[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{4}")
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_.-]+")
INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
SINHALA_RE = re.compile(r"[\u0D80-\u0DFF]")
LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_text(text: Any, *, mask_private: bool = True) -> str:
    value = unicodedata.normalize("NFC", str(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = INVISIBLE_RE.sub("", value)
    if mask_private:
        value = PHONE_RE.sub("<PHONE>", value)
        value = EMAIL_RE.sub("<EMAIL>", value)
        value = URL_RE.sub("<URL>", value)
        value = MENTION_RE.sub("<USER>", value)
    return re.sub(r"\s+", " ", value).strip()


def text_hash(text: str) -> str:
    canonical = normalize_text(text, mask_private=True).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def detect_language(text: str) -> str:
    has_sinhala = bool(SINHALA_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))
    if has_sinhala and has_latin:
        return "mixed"
    if has_sinhala:
        return "sinhala"
    if has_latin:
        return "singlish"
    return "other"


def _validate_schema(frame: pd.DataFrame) -> None:
    missing = [column for column in HEADER_MAP if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")


def _assign_group_splits(frame: pd.DataFrame, seed: int) -> pd.Series:
    """Assign complete duplicate groups to deterministic stratified splits."""
    group_rows = (
        frame.groupby("text_hash", as_index=False)
        .agg(sentiment_label=("sentiment_label", "first"), language_type=("language_type", "first"))
    )
    assignment: dict[str, str] = {}
    rng = np.random.default_rng(seed)
    for _, stratum in group_rows.groupby(["sentiment_label", "language_type"], sort=True):
        group_ids = stratum["text_hash"].tolist()
        rng.shuffle(group_ids)
        count = len(group_ids)
        train_end = round(count * 0.70)
        validation_end = train_end + round(count * 0.15)
        if count >= 3:
            train_end = max(1, min(train_end, count - 2))
            validation_end = max(train_end + 1, min(validation_end, count - 1))
        for index, group_id in enumerate(group_ids):
            if index < train_end:
                assignment[group_id] = "train"
            elif index < validation_end:
                assignment[group_id] = "validation"
            else:
                assignment[group_id] = "test"
    return frame["text_hash"].map(assignment)


def _select_annotation_candidates(frame: pd.DataFrame, seed: int, target: int = 120) -> pd.DataFrame:
    unique = frame.sort_values(
        ["label_confidence", "total_reactions"], ascending=[True, False]
    ).drop_duplicates("text_hash")
    selected_parts: list[pd.DataFrame] = []
    labels = sorted(unique["sentiment_label"].unique())
    per_label = max(1, target // max(1, len(labels)))
    for label in labels:
        subset = unique[unique["sentiment_label"] == label]
        uncertain = subset.head(math.ceil(per_label / 2))
        remaining = subset.drop(uncertain.index)
        random_part = remaining.sample(
            n=min(per_label - len(uncertain), len(remaining)), random_state=seed
        )
        selected_parts.extend([uncertain, random_part])
    candidates = pd.concat(selected_parts, ignore_index=True).drop_duplicates("record_id").head(target)
    # Keep the annotation task blinded: annotators see text and language only,
    # never reaction counts, model confidence, or the provisional algorithm label.
    columns = ["record_id", "model_text", "language_type"]
    candidates = candidates[columns].copy()
    candidates["annotator_1_label"] = ""
    candidates["annotator_2_label"] = ""
    candidates["adjudicated_label"] = ""
    candidates["annotation_notes"] = ""
    return candidates


def preprocess_dataset(
    input_path: Path,
    config_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    release_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = pd.read_excel(input_path, sheet_name="FB Posts", engine="openpyxl")
    _validate_schema(source)
    source = source[list(HEADER_MAP)].rename(columns=HEADER_MAP)

    rejection_reasons: list[str] = []
    calculated_totals: list[int] = []
    for _, row in source.iterrows():
        reasons: list[str] = []
        for column in REACTION_COLUMNS:
            value = row[column]
            if pd.isna(value) or not isinstance(value, (int, float, np.integer, np.floating)):
                reasons.append(f"invalid_{column}")
            elif float(value) < 0 or not float(value).is_integer():
                reasons.append(f"invalid_{column}")
        calculated = int(sum(float(row[column]) for column in REACTION_COLUMNS))
        calculated_totals.append(calculated)
        if calculated != int(row["source_total_reactions"]):
            reasons.append("reaction_total_mismatch")
        normalized_unmasked = normalize_text(row["raw_text"], mask_private=False)
        if not normalized_unmasked:
            reasons.append("empty_text")
        if normalized_unmasked == "232" and calculated == 0:
            reasons.append("placeholder_text_and_zero_reactions")
        elif calculated < int(config["minimum_total_reactions"]):
            reasons.append("below_minimum_reactions")
        rejection_reasons.append(";".join(sorted(set(reasons))))

    source["calculated_total_reactions"] = calculated_totals
    source["rejection_reason"] = rejection_reasons
    rejected = source[source["rejection_reason"] != ""].copy()
    clean = source[source["rejection_reason"] == ""].copy()

    for column in REACTION_COLUMNS + ("source_total_reactions", "calculated_total_reactions"):
        clean[column] = clean[column].astype("int64")
    clean["record_id"] = clean["source_row_id"].map(lambda value: f"RF_POST_{int(value):06d}")
    clean["normalized_text"] = clean["raw_text"].map(
        lambda value: normalize_text(value, mask_private=False)
    )
    clean["model_text"] = clean["raw_text"].map(normalize_text)
    clean["privacy_masked"] = clean["normalized_text"] != clean["model_text"]
    clean["language_type"] = clean["model_text"].map(detect_language)
    clean["text_hash"] = clean["model_text"].map(text_hash)
    clean["duplicate_group_id"] = clean["text_hash"].str[:16].map(lambda value: f"DUP_{value}")
    duplicate_counts = clean.groupby("text_hash")["text_hash"].transform("size")
    clean["duplicate_count"] = duplicate_counts.astype("int64")
    clean["is_duplicate"] = clean["duplicate_count"] > 1
    clean["total_reactions"] = clean[list(REACTION_COLUMNS)].sum(axis=1).astype("int64")
    clean["log_total_reactions"] = np.log1p(clean["total_reactions"])

    alpha = float(config["smoothing_alpha"])
    for reaction in REACTIONS:
        count_column = f"{reaction}_count"
        clean[f"{reaction}_ratio"] = clean[count_column] / clean["total_reactions"]
        clean[f"{reaction}_ratio_smoothed"] = (
            clean[count_column] + alpha
        ) / (clean["total_reactions"] + alpha * len(REACTIONS))

    count_matrix = clean[list(REACTION_COLUMNS)].to_numpy()
    dominant_indices = count_matrix.argmax(axis=1)
    sorted_counts = np.sort(count_matrix, axis=1)
    clean["dominant_reaction"] = [REACTIONS[index] for index in dominant_indices]
    clean["dominant_ratio"] = count_matrix.max(axis=1) / clean["total_reactions"].to_numpy()
    clean["dominance_margin"] = (
        sorted_counts[:, -1] - sorted_counts[:, -2]
    ) / clean["total_reactions"].to_numpy()

    fusion_rows: list[dict[str, Any]] = []
    for _, row in clean.iterrows():
        counts = {reaction: int(row[f"{reaction}_count"]) for reaction in REACTIONS}
        result = fuse_reactions(counts, config)
        baseline_label, baseline_score = filtered_baseline(counts)
        fusion_rows.append(
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
    clean = pd.concat([clean.reset_index(drop=True), pd.DataFrame(fusion_rows)], axis=1)
    clean["split"] = _assign_group_splits(clean, seed)
    clean["quality_status"] = np.where(clean["privacy_masked"], "approved_masked", "approved")

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(interim_dir / "facebook_posts_cleaned_private.csv", index=False, encoding="utf-8-sig")
    rejected.to_csv(interim_dir / "rejected_records_private.csv", index=False, encoding="utf-8-sig")

    public_drop = ["raw_text", "normalized_text", "rejection_reason"]
    public = clean.drop(columns=public_drop)
    public.to_csv(processed_dir / "dataset.csv", index=False, encoding="utf-8-sig")
    deduplicated = public.sort_values(
        ["text_hash", "total_reactions"], ascending=[True, False]
    ).drop_duplicates("text_hash")
    deduplicated.to_csv(processed_dir / "dataset_deduplicated.csv", index=False, encoding="utf-8-sig")
    for split in ("train", "validation", "test"):
        public[public["split"] == split].to_csv(
            processed_dir / f"{split}.csv", index=False, encoding="utf-8-sig"
        )

    annotation_candidates = _select_annotation_candidates(public, seed)
    annotation_candidates.to_csv(
        processed_dir / "human_annotation_candidates.csv", index=False, encoding="utf-8-sig"
    )

    report: dict[str, Any] = {
        "source_records": int(len(source)),
        "accepted_records": int(len(public)),
        "rejected_records": int(len(rejected)),
        "privacy_masked_records": int(public["privacy_masked"].sum()),
        "duplicate_groups": int((public.groupby("text_hash").size() > 1).sum()),
        "duplicate_occurrences_beyond_first": int(len(public) - public["text_hash"].nunique()),
        "deduplicated_records": int(len(deduplicated)),
        "language_distribution": public["language_type"].value_counts().sort_index().to_dict(),
        "label_distribution": public["sentiment_label"].value_counts().sort_index().to_dict(),
        "baseline_label_distribution": public["baseline_filtered_label"].value_counts().sort_index().to_dict(),
        "split_distribution": public["split"].value_counts().sort_index().to_dict(),
        "split_label_distribution": pd.crosstab(
            public["split"], public["sentiment_label"]
        ).to_dict(orient="index"),
        "low_confidence_records": int(
            (public["label_confidence"] < float(config["low_confidence_threshold"])).sum()
        ),
        "annotation_candidates": int(len(annotation_candidates)),
        "label_version": config["version"],
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    (processed_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (processed_dir / "labeling_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    release_files = [
        "dataset.csv",
        "dataset_deduplicated.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
        "human_annotation_candidates.csv",
        "quality_report.json",
        "labeling_config.json",
    ]
    for filename in release_files:
        (release_dir / filename).write_bytes((processed_dir / filename).read_bytes())
    rejected_public = rejected.drop(columns=["raw_text"], errors="ignore")
    rejected_public.to_csv(release_dir / "rejected_records.csv", index=False, encoding="utf-8-sig")
    return report


def _write_dataset_card(release_dir: Path, report: dict[str, Any]) -> None:
    card = f"""# ReactionFusion v1 provisional dataset card

## Status

This is a privacy-masked research release generated from 1,000 collected Facebook
post records. ReactionFusion labels are **provisional weak labels** and must not be
treated as human ground truth. Complete the included annotation task before final
algorithm validation or benchmark reporting.

## Processing summary

- Source records: {report['source_records']}
- Accepted records: {report['accepted_records']}
- Rejected records: {report['rejected_records']}
- Records with privacy masking: {report['privacy_masked_records']}
- Duplicate groups: {report['duplicate_groups']}
- Extra duplicate occurrences: {report['duplicate_occurrences_beyond_first']}
- Deduplicated records: {report['deduplicated_records']}
- Human annotation candidates: {report['annotation_candidates']}
- Label version: `{report['label_version']}`

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
"""
    (release_dir / "DATASET_CARD.md").write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/source_exports/facebook_posts_2026-08-14.xlsx"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/labeling/reactionfusion_v1.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    version = "reactionfusion_v1"
    release_dir = Path("data/releases") / version
    report = preprocess_dataset(
        input_path=args.input,
        config_path=args.config,
        interim_dir=Path("data/interim"),
        processed_dir=Path("data/processed") / version,
        release_dir=release_dir,
        seed=args.seed,
    )
    _write_dataset_card(release_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
