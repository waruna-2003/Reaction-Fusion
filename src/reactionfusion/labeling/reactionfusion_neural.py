"""Small-data multi-task neural ReactionFusion model implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from reactionfusion.labeling.reactionfusion_v2 import Standardizer


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(np.clip(shifted, -50.0, 50.0))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class NeuralMember:
    input_weights: np.ndarray
    input_bias: np.ndarray
    sentiment_weights: np.ndarray
    sentiment_bias: np.ndarray
    emotion_weights: np.ndarray
    emotion_bias: np.ndarray
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_weights": self.input_weights.tolist(),
            "input_bias": self.input_bias.tolist(),
            "sentiment_weights": self.sentiment_weights.tolist(),
            "sentiment_bias": self.sentiment_bias.tolist(),
            "emotion_weights": self.emotion_weights.tolist(),
            "emotion_bias": self.emotion_bias.tolist(),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "NeuralMember":
        return cls(
            input_weights=np.asarray(values["input_weights"], dtype=float),
            input_bias=np.asarray(values["input_bias"], dtype=float),
            sentiment_weights=np.asarray(values["sentiment_weights"], dtype=float),
            sentiment_bias=np.asarray(values["sentiment_bias"], dtype=float),
            emotion_weights=np.asarray(values["emotion_weights"], dtype=float),
            emotion_bias=np.asarray(values["emotion_bias"], dtype=float),
            seed=int(values["seed"]),
        )


@dataclass(frozen=True)
class ReactionFusionNeuralModel:
    version: str
    feature_names: tuple[str, ...]
    sentiment_classes: tuple[str, ...]
    emotion_targets: tuple[str, ...]
    standardizer: Standardizer
    members: tuple[NeuralMember, ...]
    temperature: float
    abstention_threshold: float
    smoothing_alpha: float
    metadata: Mapping[str, Any]

    def predict_probability(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        standardized = self.standardizer.transform(features)
        sentiments = []
        emotions = []
        for member in self.members:
            hidden = np.tanh(standardized @ member.input_weights + member.input_bias)
            sentiments.append(_softmax(hidden @ member.sentiment_weights + member.sentiment_bias))
            emotions.append(_sigmoid(hidden @ member.emotion_weights + member.emotion_bias))
        sentiment = np.mean(sentiments, axis=0)
        emotion = np.mean(emotions, axis=0)
        logits = np.log(np.clip(sentiment, 1e-12, 1.0)) / max(self.temperature, 1e-6)
        return _softmax(logits), emotion

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "sentiment_classes": list(self.sentiment_classes),
            "emotion_targets": list(self.emotion_targets),
            "standardizer": self.standardizer.to_dict(),
            "members": [member.to_dict() for member in self.members],
            "temperature": self.temperature,
            "abstention_threshold": self.abstention_threshold,
            "smoothing_alpha": self.smoothing_alpha,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ReactionFusionNeuralModel":
        return cls(
            version=str(values["version"]),
            feature_names=tuple(values["feature_names"]),
            sentiment_classes=tuple(values["sentiment_classes"]),
            emotion_targets=tuple(values["emotion_targets"]),
            standardizer=Standardizer.from_dict(values["standardizer"]),
            members=tuple(NeuralMember.from_dict(item) for item in values["members"]),
            temperature=float(values["temperature"]),
            abstention_threshold=float(values["abstention_threshold"]),
            smoothing_alpha=float(values["smoothing_alpha"]),
            metadata=values.get("metadata", {}),
        )


def _balanced_weights(
    target: np.ndarray, mask: np.ndarray, classes: int, power: float = 0.5
) -> np.ndarray:
    weights = np.ones(len(target), dtype=float)
    counts = np.bincount(target[mask], minlength=classes).astype(float)
    nonzero = counts[counts > 0]
    if not len(nonzero):
        return weights
    reference = float(nonzero.mean())
    class_weights = np.power(reference / np.maximum(counts, 1.0), power)
    class_weights = np.clip(class_weights, 0.5, 2.5)
    weights[mask] = class_weights[target[mask]]
    weights[~mask] = 0.0
    return weights


def _adam_update(
    parameter: np.ndarray,
    gradient: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    step: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = 0.9 * first + 0.1 * gradient
    second = 0.999 * second + 0.001 * gradient * gradient
    corrected_first = first / (1.0 - 0.9**step)
    corrected_second = second / (1.0 - 0.999**step)
    parameter = parameter - learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return parameter, first, second


def _train_member(
    features: np.ndarray,
    annotations: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
    initial_member: NeuralMember | None = None,
) -> NeuralMember:
    rng = np.random.default_rng(seed)
    classes = tuple(config["sentiment_classes"])
    emotions = tuple(config["emotion_targets"])
    class_to_id = {label: index for index, label in enumerate(classes)}
    sentiment_text = annotations["sentiment"].astype("string").str.strip().str.lower()
    sentiment_mask = sentiment_text.isin(classes).to_numpy()
    sentiment_ids = np.asarray(
        [class_to_id.get(str(value), 0) for value in sentiment_text], dtype=int
    )
    class_weight_power = float(config.get("class_weight_power", 0.5))
    sentiment_weights = _balanced_weights(
        sentiment_ids, sentiment_mask, len(classes), class_weight_power
    )
    row_weights = np.ones(len(annotations), dtype=float)
    if "sample_weight" in annotations:
        row_weights = pd.to_numeric(annotations["sample_weight"], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.all(np.isfinite(row_weights)) or np.any(row_weights <= 0):
            raise ValueError("Neural sample_weight values must be finite and positive")
    sentiment_weights *= row_weights

    emotion_targets = np.zeros((len(annotations), len(emotions)), dtype=float)
    emotion_masks = np.zeros_like(emotion_targets, dtype=bool)
    emotion_weights = np.zeros_like(emotion_targets, dtype=float)
    for index, emotion in enumerate(emotions):
        values = annotations[emotion].astype("string").str.strip().str.lower()
        mask = values.isin({"yes", "no"}).to_numpy()
        target = (values == "yes").to_numpy(dtype=int)
        emotion_targets[:, index] = target
        emotion_masks[:, index] = mask
        emotion_weights[:, index] = _balanced_weights(
            target, mask, 2, class_weight_power
        )
    emotion_weights *= row_weights[:, None]

    rows, inputs = features.shape
    hidden_units = int(config["hidden_units"])
    if initial_member is None:
        input_weights = rng.normal(0.0, np.sqrt(1.0 / inputs), (inputs, hidden_units))
        input_bias = np.zeros(hidden_units)
        output_scale = np.sqrt(1.0 / hidden_units)
        sentiment_output = rng.normal(0.0, output_scale, (hidden_units, len(classes)))
        sentiment_bias = np.zeros(len(classes))
        emotion_output = rng.normal(0.0, output_scale, (hidden_units, len(emotions)))
        emotion_bias = np.zeros(len(emotions))
    else:
        expected_shapes = (
            (inputs, hidden_units),
            (hidden_units,),
            (hidden_units, len(classes)),
            (len(classes),),
            (hidden_units, len(emotions)),
            (len(emotions),),
        )
        existing = (
            initial_member.input_weights,
            initial_member.input_bias,
            initial_member.sentiment_weights,
            initial_member.sentiment_bias,
            initial_member.emotion_weights,
            initial_member.emotion_bias,
        )
        if tuple(value.shape for value in existing) != expected_shapes:
            raise ValueError("Initial neural member architecture does not match fine-tuning config")
        (
            input_weights,
            input_bias,
            sentiment_output,
            sentiment_bias,
            emotion_output,
            emotion_bias,
        ) = (value.copy() for value in existing)
    parameters = [
        input_weights,
        input_bias,
        sentiment_output,
        sentiment_bias,
        emotion_output,
        emotion_bias,
    ]
    first_moments = [np.zeros_like(parameter) for parameter in parameters]
    second_moments = [np.zeros_like(parameter) for parameter in parameters]
    smoothing = float(config["label_smoothing"])
    emotion_loss_weight = float(config["emotion_loss_weight"])
    l2 = float(config["l2"])
    noise = float(config.get("feature_noise_std", 0.0))

    targets = np.full((rows, len(classes)), smoothing / len(classes))
    targets[np.arange(rows), sentiment_ids] += 1.0 - smoothing
    sentiment_denominator = max(float(sentiment_weights.sum()), 1.0)
    emotion_denominators = np.maximum(emotion_weights.sum(axis=0), 1.0)

    for step in range(1, int(config["epochs"]) + 1):
        network_input = features
        if noise > 0:
            network_input = features + rng.normal(0.0, noise, features.shape)
        hidden = np.tanh(network_input @ parameters[0] + parameters[1])
        sentiment_probability = _softmax(hidden @ parameters[2] + parameters[3])
        emotion_probability = _sigmoid(hidden @ parameters[4] + parameters[5])

        sentiment_gradient = (sentiment_probability - targets) * sentiment_weights[:, None]
        sentiment_gradient /= sentiment_denominator
        emotion_gradient = (emotion_probability - emotion_targets) * emotion_weights
        emotion_gradient /= emotion_denominators[None, :]
        emotion_gradient *= emotion_loss_weight / len(emotions)

        gradient_sentiment_output = hidden.T @ sentiment_gradient + l2 * parameters[2]
        gradient_sentiment_bias = sentiment_gradient.sum(axis=0)
        gradient_emotion_output = hidden.T @ emotion_gradient + l2 * parameters[4]
        gradient_emotion_bias = emotion_gradient.sum(axis=0)
        hidden_gradient = sentiment_gradient @ parameters[2].T
        hidden_gradient += emotion_gradient @ parameters[4].T
        activation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        gradient_input_weights = network_input.T @ activation_gradient + l2 * parameters[0]
        gradient_input_bias = activation_gradient.sum(axis=0)
        gradients = [
            gradient_input_weights,
            gradient_input_bias,
            gradient_sentiment_output,
            gradient_sentiment_bias,
            gradient_emotion_output,
            gradient_emotion_bias,
        ]
        for index, gradient in enumerate(gradients):
            parameters[index], first_moments[index], second_moments[index] = _adam_update(
                parameters[index],
                gradient,
                first_moments[index],
                second_moments[index],
                step,
                float(config["learning_rate"]),
            )

    return NeuralMember(
        input_weights=parameters[0],
        input_bias=parameters[1],
        sentiment_weights=parameters[2],
        sentiment_bias=parameters[3],
        emotion_weights=parameters[4],
        emotion_bias=parameters[5],
        seed=seed,
    )


def train_neural_model(
    features: np.ndarray,
    annotations: pd.DataFrame,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> ReactionFusionNeuralModel:
    """Train a deterministic ensemble of regularized multi-task neural networks."""
    sample_weight = None
    if "sample_weight" in annotations:
        sample_weight = pd.to_numeric(
            annotations["sample_weight"], errors="coerce"
        ).to_numpy(dtype=float)
    standardizer = Standardizer.fit(features, sample_weight=sample_weight)
    standardized = standardizer.transform(features)
    members = tuple(
        _train_member(standardized, annotations, config, int(seed))
        for seed in config["ensemble_seeds"]
    )
    return ReactionFusionNeuralModel(
        version=str(config["version"]),
        feature_names=tuple(feature_names),
        sentiment_classes=tuple(config["sentiment_classes"]),
        emotion_targets=tuple(config["emotion_targets"]),
        standardizer=standardizer,
        members=members,
        temperature=1.0,
        abstention_threshold=float(config["minimum_abstention_threshold"]),
        smoothing_alpha=float(config["smoothing_alpha"]),
        metadata=metadata or {},
    )


def fine_tune_neural_model(
    base_model: ReactionFusionNeuralModel,
    features: np.ndarray,
    annotations: pd.DataFrame,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> ReactionFusionNeuralModel:
    """Fine-tune every pretrained ensemble member without refitting its input scale."""
    if tuple(config["sentiment_classes"]) != base_model.sentiment_classes:
        raise ValueError("Fine-tuning sentiment classes do not match the pretrained model")
    if tuple(config["emotion_targets"]) != base_model.emotion_targets:
        raise ValueError("Fine-tuning emotion targets do not match the pretrained model")
    if int(config["hidden_units"]) != base_model.members[0].input_bias.shape[0]:
        raise ValueError("Fine-tuning hidden-unit count does not match the pretrained model")
    target_standardizer = base_model.standardizer
    initial_members = base_model.members
    if bool(config.get("refit_standardizer", False)):
        target_standardizer = Standardizer.fit(features)
        scale_ratio = target_standardizer.scale / base_model.standardizer.scale
        mean_shift = (
            target_standardizer.mean - base_model.standardizer.mean
        ) / base_model.standardizer.scale
        adapted_members = []
        for member in initial_members:
            adapted_members.append(
                NeuralMember(
                    input_weights=member.input_weights * scale_ratio[:, None],
                    input_bias=member.input_bias + mean_shift @ member.input_weights,
                    sentiment_weights=member.sentiment_weights.copy(),
                    sentiment_bias=member.sentiment_bias.copy(),
                    emotion_weights=member.emotion_weights.copy(),
                    emotion_bias=member.emotion_bias.copy(),
                    seed=member.seed,
                )
            )
        initial_members = tuple(adapted_members)
    standardized = target_standardizer.transform(features)
    if bool(config.get("reset_output_heads", False)):
        reset_members = []
        hidden_units = int(config["hidden_units"])
        output_scale = np.sqrt(1.0 / hidden_units)
        for member in base_model.members:
            rng = np.random.default_rng(member.seed + 10_000)
            reset_members.append(
                NeuralMember(
                    input_weights=member.input_weights.copy(),
                    input_bias=member.input_bias.copy(),
                    sentiment_weights=rng.normal(
                        0.0,
                        output_scale,
                        (hidden_units, len(base_model.sentiment_classes)),
                    ),
                    sentiment_bias=np.zeros(len(base_model.sentiment_classes)),
                    emotion_weights=rng.normal(
                        0.0,
                        output_scale,
                        (hidden_units, len(base_model.emotion_targets)),
                    ),
                    emotion_bias=np.zeros(len(base_model.emotion_targets)),
                    seed=member.seed,
                )
            )
        initial_members = tuple(reset_members)
    members = tuple(
        _train_member(
            standardized,
            annotations,
            config,
            member.seed,
            initial_member=member,
        )
        for member in initial_members
    )
    return ReactionFusionNeuralModel(
        version=str(config["version"]),
        feature_names=base_model.feature_names,
        sentiment_classes=base_model.sentiment_classes,
        emotion_targets=base_model.emotion_targets,
        standardizer=target_standardizer,
        members=members,
        temperature=1.0,
        abstention_threshold=float(config["minimum_abstention_threshold"]),
        smoothing_alpha=base_model.smoothing_alpha,
        metadata=metadata or {},
    )
