import numpy as np
import pandas as pd

from reactionfusion.labeling.reactionfusion_v2 import (
    BASE_FEATURE_NAMES,
    ReactionFusionV2Model,
    extract_reaction_features,
    predict_v2_matrix,
    train_v2_model,
)
from reactionfusion.training.reactionfusion_v2_pipeline import assign_group_folds


def test_v2_feature_engine_uses_all_reactions() -> None:
    counts = {"like": 10, "love": 4, "care": 3, "haha": 5, "wow": 2, "sad": 6, "angry": 1}
    features, audit = extract_reaction_features(counts)
    assert len(features) == len(BASE_FEATURE_NAMES)
    assert np.isfinite(features).all()
    assert audit["total_reactions"] == 31
    assert audit["mixed_evidence"] > 0


def test_v2_rejects_zero_reactions() -> None:
    try:
        extract_reaction_features(
            {
                reaction: 0
                for reaction in ["like", "love", "care", "haha", "wow", "sad", "angry"]
            }
        )
    except ValueError as error:
        assert "at least one reaction" in str(error)
    else:
        raise AssertionError("Zero-reaction input should fail")


def test_group_folds_never_split_a_text_group() -> None:
    labels = pd.Series(["negative", "negative", "positive", "positive", "mixed", "neutral"])
    groups = pd.Series(["same", "same", "p1", "p2", "m1", "n1"])
    folds = assign_group_folds(labels, groups, n_splits=2, seed=42)
    assert folds[0] == folds[1]
    assert set(folds) == {0, 1}


def test_v2_model_round_trip() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(8, len(BASE_FEATURE_NAMES)))
    emotions = ["joy", "sadness"]
    annotations = pd.DataFrame(
        {
            "sentiment": ["negative", "neutral", "positive", "mixed"] * 2,
            "joy": ["no", "no", "yes", "yes", "no", "yes", "yes", "no"],
            "sadness": ["yes", "no", "no", "yes", "yes", "no", "no", "yes"],
        }
    )
    config = {
        "version": "test_v2",
        "emotion_targets": emotions,
        "sentiment_classes": ["negative", "neutral", "positive", "mixed"],
        "binary_iterations": 10,
        "binary_learning_rate": 0.02,
        "binary_l2": 0.1,
        "sentiment_iterations": 10,
        "sentiment_learning_rate": 0.02,
        "sentiment_l2": 0.1,
        "minimum_abstention_threshold": 0.3,
        "smoothing_alpha": 1.0,
    }
    model = train_v2_model(features, annotations, config)
    restored = ReactionFusionV2Model.from_dict(model.to_dict())
    original_probabilities, original_emotions = predict_v2_matrix(model, features)
    restored_probabilities, restored_emotions = predict_v2_matrix(restored, features)
    assert np.allclose(original_probabilities, restored_probabilities)
    assert np.allclose(original_emotions, restored_emotions)
