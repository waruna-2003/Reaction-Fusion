import json
from pathlib import Path

from reactionfusion.labeling import filtered_baseline, fuse_reactions


CONFIG = json.loads(Path("configs/labeling/reactionfusion_v1.json").read_text(encoding="utf-8"))


def test_clear_positive_context_makes_ambiguous_reactions_positive() -> None:
    result = fuse_reactions(
        {"like": 50, "love": 30, "care": 5, "haha": 10, "wow": 2, "sad": 0, "angry": 0},
        CONFIG,
    )
    assert result.sentiment_label == "positive"
    assert result.fusion_score > 0


def test_clear_negative_context_makes_haha_context_negative() -> None:
    result = fuse_reactions(
        {"like": 5, "love": 0, "care": 0, "haha": 40, "wow": 2, "sad": 20, "angry": 10},
        CONFIG,
    )
    assert result.sentiment_label == "negative"
    assert result.fusion_score < 0


def test_filtered_baseline_discards_ambiguous_reactions() -> None:
    label, score = filtered_baseline({"like": 100, "haha": 100, "love": 0, "care": 0, "sad": 0, "angry": 0})
    assert label == "neutral"
    assert score == 0
