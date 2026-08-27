"""Tests for provenance-safe augmented-data merging."""

import pandas as pd

from reactionfusion.data.merge_augmented import _attach_annotations
from reactionfusion.evaluation.human_validation import BINARY_COLUMNS


def _annotations(record_id: str, sentiment: str) -> pd.DataFrame:
    values = {
        "record_id": [record_id],
        "sentiment": [sentiment],
        "confidence": ["high"],
    }
    values.update({column: ["no"] for column in BINARY_COLUMNS})
    return pd.DataFrame(values)


def test_annotation_sources_remain_distinguishable() -> None:
    combined = pd.DataFrame(
        {
            "record_id": ["RF_POST_000001", "SYN_POST_000001", "RF_POST_000002"],
            "data_origin": ["legacy_source", "synthetic_augmentation", "legacy_source"],
        }
    )
    result = _attach_annotations(
        combined,
        _annotations("SYN_POST_000001", "mixed"),
        _annotations("RF_POST_000001", "positive"),
    ).set_index("record_id")
    assert result.loc["RF_POST_000001", "annotation_provenance"] == (
        "human_adjudicated_development"
    )
    assert result.loc["SYN_POST_000001", "annotation_provenance"] == (
        "provided_synthetic_unverified"
    )
    assert result.loc["RF_POST_000002", "annotation_provenance"] == "none"
    assert result.loc["SYN_POST_000001", "provided_sentiment"] == "mixed"


def test_annotation_id_collision_is_rejected() -> None:
    combined = pd.DataFrame({"record_id": ["DUPLICATE"]})
    annotations = _annotations("DUPLICATE", "neutral")
    try:
        _attach_annotations(combined, annotations, annotations)
    except ValueError as error:
        assert "collide" in str(error)
    else:
        raise AssertionError("Expected duplicate annotation IDs to be rejected")
