"""Unit tests for provenance-weighted combined neural training."""

import pandas as pd

from reactionfusion.evaluation.human_validation import BINARY_COLUMNS
from reactionfusion.training.reactionfusion_combined_pipeline import _annotation_frame


def test_annotation_frame_applies_provenance_weight() -> None:
    values = {
        "record_id": ["SYN_POST_000001"],
        "provided_sentiment": ["mixed"],
    }
    values.update({f"provided_{column}": ["no"] for column in BINARY_COLUMNS})
    annotations = _annotation_frame(pd.DataFrame(values), weight=0.0001)
    assert annotations.loc[0, "sentiment"] == "mixed"
    assert annotations.loc[0, "sample_weight"] == 0.0001
    assert "provided_sentiment" not in annotations
