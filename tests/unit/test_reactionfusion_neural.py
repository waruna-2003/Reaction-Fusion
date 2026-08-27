"""Unit tests for the NumPy ReactionFusion neural model."""

import numpy as np
import pandas as pd

from reactionfusion.labeling.reactionfusion_neural import (
    ReactionFusionNeuralModel,
    fine_tune_neural_model,
    train_neural_model,
)
from reactionfusion.labeling.reactionfusion_v2 import Standardizer


def _config() -> dict:
    return {
        "version": "test_neural",
        "sentiment_classes": ["negative", "neutral", "positive", "mixed"],
        "emotion_targets": ["joy", "sarcasm"],
        "hidden_units": 5,
        "ensemble_seeds": [3, 7],
        "epochs": 30,
        "learning_rate": 0.01,
        "l2": 0.01,
        "label_smoothing": 0.05,
        "emotion_loss_weight": 0.3,
        "feature_noise_std": 0.0,
        "smoothing_alpha": 1.0,
        "minimum_abstention_threshold": 0.28,
    }


def _training_data() -> tuple[np.ndarray, pd.DataFrame]:
    features = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4],
            [0.5, 0.0, 0.5],
            [0.4, 0.2, 0.4],
        ]
    )
    annotations = pd.DataFrame(
        {
            "sentiment": [
                "positive",
                "positive",
                "negative",
                "negative",
                "neutral",
                "neutral",
                "mixed",
                "mixed",
            ],
            "joy": ["yes", "yes", "no", "no", "no", "no", "yes", "uncertain"],
            "sarcasm": ["no", "no", "no", "no", "no", "yes", "yes", "yes"],
        }
    )
    return features, annotations


def test_neural_probabilities_are_valid() -> None:
    features, annotations = _training_data()
    model = train_neural_model(features, annotations, ("a", "b", "c"), _config())
    sentiment, emotion = model.predict_probability(features)
    assert sentiment.shape == (8, 4)
    assert emotion.shape == (8, 2)
    assert np.allclose(sentiment.sum(axis=1), 1.0)
    assert np.all((emotion >= 0.0) & (emotion <= 1.0))


def test_neural_model_round_trip_is_exact() -> None:
    features, annotations = _training_data()
    model = train_neural_model(features, annotations, ("a", "b", "c"), _config())
    restored = ReactionFusionNeuralModel.from_dict(model.to_dict())
    original = model.predict_probability(features)
    recovered = restored.predict_probability(features)
    assert np.allclose(original[0], recovered[0])
    assert np.allclose(original[1], recovered[1])


def test_neural_training_is_deterministic() -> None:
    features, annotations = _training_data()
    first = train_neural_model(features, annotations, ("a", "b", "c"), _config())
    second = train_neural_model(features, annotations, ("a", "b", "c"), _config())
    assert np.allclose(
        first.predict_probability(features)[0], second.predict_probability(features)[0]
    )


def test_weighted_standardizer_respects_row_provenance() -> None:
    values = np.asarray([[0.0], [10.0]])
    standardizer = Standardizer.fit(values, sample_weight=np.asarray([1.0, 0.1]))
    assert standardizer.mean[0] < 1.0


def test_pretrained_model_can_be_fine_tuned() -> None:
    features, annotations = _training_data()
    base = train_neural_model(features, annotations, ("a", "b", "c"), _config())
    fine_config = {**_config(), "version": "fine_tuned_test", "epochs": 10}
    fine_tuned = fine_tune_neural_model(
        base, features, annotations, fine_config, metadata={"stage": "fine_tune"}
    )
    sentiment, emotion = fine_tuned.predict_probability(features)
    assert fine_tuned.version == "fine_tuned_test"
    assert sentiment.shape == (8, 4)
    assert emotion.shape == (8, 2)
