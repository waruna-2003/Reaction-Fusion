"""Human-calibrated, reaction-only ReactionFusion v2 model.

Small regularized linear models are used because the available development set
contains only 120 uncertainty-enriched human annotations. Text is never an input,
which prevents circular label generation for downstream text classifiers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REACTIONS = ("like", "love", "care", "haha", "wow", "sad", "angry")
BASE_FEATURE_NAMES = (
    *(f"ratio_{reaction}" for reaction in REACTIONS),
    "log_engagement",
    "reaction_entropy",
    "dominance_margin",
    "positive_anchor",
    "negative_anchor",
    "anchor_balance",
    "opposition",
    "anchor_context",
    "like_context_positive",
    "like_context_negative",
    "haha_context_positive",
    "haha_context_negative",
    "wow_context_positive",
    "wow_context_negative",
    "care_sad_interaction",
    "care_angry_interaction",
    "love_sad_interaction",
    "love_angry_interaction",
    "mixed_evidence",
    "ambiguous_mass",
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(np.clip(shifted, -50.0, 50.0))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        return cls(mean=mean, scale=np.where(scale < 1e-9, 1.0, scale))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "Standardizer":
        return cls(
            mean=np.asarray(values["mean"], dtype=float),
            scale=np.asarray(values["scale"], dtype=float),
        )


@dataclass(frozen=True)
class BinaryLogisticModel:
    standardizer: Standardizer
    coefficients: np.ndarray
    intercept: float
    constant_probability: float | None = None

    def predict_probability(self, values: np.ndarray) -> np.ndarray:
        if self.constant_probability is not None:
            return np.full(len(values), self.constant_probability, dtype=float)
        standardized = self.standardizer.transform(values)
        return _sigmoid(standardized @ self.coefficients + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "standardizer": self.standardizer.to_dict(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "constant_probability": self.constant_probability,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "BinaryLogisticModel":
        constant = values.get("constant_probability")
        return cls(
            standardizer=Standardizer.from_dict(values["standardizer"]),
            coefficients=np.asarray(values["coefficients"], dtype=float),
            intercept=float(values["intercept"]),
            constant_probability=None if constant is None else float(constant),
        )


@dataclass(frozen=True)
class SoftmaxModel:
    classes: tuple[str, ...]
    standardizer: Standardizer
    coefficients: np.ndarray
    intercepts: np.ndarray

    def predict_probability(self, values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        standardized = self.standardizer.transform(values)
        logits = standardized @ self.coefficients + self.intercepts
        return _softmax(logits / max(float(temperature), 1e-6))

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "standardizer": self.standardizer.to_dict(),
            "coefficients": self.coefficients.tolist(),
            "intercepts": self.intercepts.tolist(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SoftmaxModel":
        return cls(
            classes=tuple(values["classes"]),
            standardizer=Standardizer.from_dict(values["standardizer"]),
            coefficients=np.asarray(values["coefficients"], dtype=float),
            intercepts=np.asarray(values["intercepts"], dtype=float),
        )


@dataclass(frozen=True)
class ReactionFusionV2Model:
    version: str
    base_feature_names: tuple[str, ...]
    emotion_targets: tuple[str, ...]
    emotion_models: Mapping[str, BinaryLogisticModel]
    sentiment_model: SoftmaxModel
    temperature: float
    abstention_threshold: float
    smoothing_alpha: float
    metadata: Mapping[str, Any]

    @property
    def sentiment_feature_names(self) -> tuple[str, ...]:
        return (
            *self.base_feature_names,
            *(f"emotion_probability_{name}" for name in self.emotion_targets),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "base_feature_names": list(self.base_feature_names),
            "emotion_targets": list(self.emotion_targets),
            "emotion_models": {
                name: model.to_dict() for name, model in self.emotion_models.items()
            },
            "sentiment_model": self.sentiment_model.to_dict(),
            "temperature": self.temperature,
            "abstention_threshold": self.abstention_threshold,
            "smoothing_alpha": self.smoothing_alpha,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ReactionFusionV2Model":
        return cls(
            version=str(values["version"]),
            base_feature_names=tuple(values["base_feature_names"]),
            emotion_targets=tuple(values["emotion_targets"]),
            emotion_models={
                name: BinaryLogisticModel.from_dict(model)
                for name, model in values["emotion_models"].items()
            },
            sentiment_model=SoftmaxModel.from_dict(values["sentiment_model"]),
            temperature=float(values["temperature"]),
            abstention_threshold=float(values["abstention_threshold"]),
            smoothing_alpha=float(values["smoothing_alpha"]),
            metadata=values.get("metadata", {}),
        )


@dataclass(frozen=True)
class ReactionFusionV2Result:
    sentiment_label: str
    candidate_sentiment_label: str
    label_confidence: float
    is_uncertain: bool
    sentiment_probabilities: Mapping[str, float]
    emotion_probabilities: Mapping[str, float]
    reaction_entropy: float
    mixed_evidence: float
    decision_reason: str


def extract_reaction_features(
    counts: Mapping[str, int | float], smoothing_alpha: float = 1.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Create auditable distribution and context-interaction features."""
    raw = np.asarray([float(counts.get(reaction, 0.0)) for reaction in REACTIONS], dtype=float)
    if np.any(raw < 0) or not np.all(np.isfinite(raw)):
        raise ValueError("Reaction counts must be finite and non-negative")
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("ReactionFusion v2 requires at least one reaction")

    proportions = (raw + smoothing_alpha) / (total + smoothing_alpha * len(REACTIONS))
    mapping = dict(zip(REACTIONS, proportions, strict=True))
    entropy = -float(
        sum(value * math.log(value) for value in proportions if value > 0)
    ) / math.log(len(REACTIONS))
    sorted_proportions = np.sort(proportions)
    dominance_margin = float(sorted_proportions[-1] - sorted_proportions[-2])
    positive_anchor = mapping["love"] + 0.55 * mapping["care"]
    negative_anchor = mapping["sad"] + mapping["angry"]
    anchor_balance = positive_anchor - negative_anchor
    opposition = min(positive_anchor, negative_anchor)
    context = math.tanh(4.0 * anchor_balance)
    positive_context = max(context, 0.0)
    negative_context = max(-context, 0.0)
    ambiguous_mass = mapping["like"] + mapping["haha"] + mapping["wow"]
    mixed_evidence = min(1.0, 4.0 * opposition * entropy)

    features = np.asarray(
        [
            *proportions.tolist(),
            math.log1p(total),
            entropy,
            dominance_margin,
            positive_anchor,
            negative_anchor,
            anchor_balance,
            opposition,
            context,
            mapping["like"] * positive_context,
            mapping["like"] * negative_context,
            mapping["haha"] * positive_context,
            mapping["haha"] * negative_context,
            mapping["wow"] * positive_context,
            mapping["wow"] * negative_context,
            mapping["care"] * mapping["sad"],
            mapping["care"] * mapping["angry"],
            mapping["love"] * mapping["sad"],
            mapping["love"] * mapping["angry"],
            mixed_evidence,
            ambiguous_mass,
        ],
        dtype=float,
    )
    audit = {
        "total_reactions": total,
        "reaction_entropy": entropy,
        "positive_anchor": positive_anchor,
        "negative_anchor": negative_anchor,
        "anchor_balance": anchor_balance,
        "opposition": opposition,
        "mixed_evidence": mixed_evidence,
        "ambiguous_mass": ambiguous_mass,
    }
    return features, audit


def feature_matrix(frame: pd.DataFrame, smoothing_alpha: float = 1.0) -> np.ndarray:
    rows = []
    for _, row in frame.iterrows():
        counts = {reaction: row[f"{reaction}_count"] for reaction in REACTIONS}
        features, _ = extract_reaction_features(counts, smoothing_alpha)
        rows.append(features)
    return np.vstack(rows)


def fit_binary_logistic(
    values: np.ndarray,
    targets: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> BinaryLogisticModel:
    standardizer = Standardizer.fit(values)
    standardized = standardizer.transform(values)
    targets = np.asarray(targets, dtype=float)
    if len(np.unique(targets)) < 2:
        constant = float((targets.sum() + 1.0) / (len(targets) + 2.0))
        return BinaryLogisticModel(
            standardizer, np.zeros(values.shape[1]), 0.0, constant
        )

    positive = max(float(targets.sum()), 1.0)
    negative = max(float(len(targets) - targets.sum()), 1.0)
    sample_weights = np.where(
        targets == 1.0,
        len(targets) / (2.0 * positive),
        len(targets) / (2.0 * negative),
    )
    coefficients = np.zeros(values.shape[1], dtype=float)
    intercept = math.log((positive + 0.5) / (negative + 0.5))
    for _ in range(iterations):
        probabilities = _sigmoid(standardized @ coefficients + intercept)
        errors = (probabilities - targets) * sample_weights
        gradient = standardized.T @ errors / len(targets)
        gradient += l2 * coefficients / len(targets)
        coefficients -= learning_rate * gradient
        intercept -= learning_rate * float(errors.mean())
    return BinaryLogisticModel(standardizer, coefficients, intercept)


def fit_softmax(
    values: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> SoftmaxModel:
    class_tuple = tuple(classes)
    class_to_id = {label: index for index, label in enumerate(class_tuple)}
    missing = [label for label in class_tuple if label not in set(labels)]
    if missing:
        raise ValueError(f"Cannot fit sentiment model without examples for: {missing}")
    target_ids = np.asarray([class_to_id[label] for label in labels], dtype=int)
    standardizer = Standardizer.fit(values)
    standardized = standardizer.transform(values)
    coefficients = np.zeros((values.shape[1], len(class_tuple)), dtype=float)
    counts = np.bincount(target_ids, minlength=len(class_tuple)).astype(float)
    intercepts = np.log((counts + 0.5) / (counts.sum() + 0.5 * len(class_tuple)))
    sample_weights = len(labels) / (len(class_tuple) * counts[target_ids])
    one_hot = np.eye(len(class_tuple))[target_ids]
    for _ in range(iterations):
        probabilities = _softmax(standardized @ coefficients + intercepts)
        errors = (probabilities - one_hot) * sample_weights[:, None]
        gradient = standardized.T @ errors / len(labels)
        gradient += l2 * coefficients / len(labels)
        coefficients -= learning_rate * gradient
        intercepts -= learning_rate * errors.mean(axis=0)
    return SoftmaxModel(class_tuple, standardizer, coefficients, intercepts)


def train_v2_model(
    base_features: np.ndarray,
    annotations: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ReactionFusionV2Model:
    emotion_targets = tuple(config["emotion_targets"])
    emotion_models: dict[str, BinaryLogisticModel] = {}
    emotion_probabilities = []
    for target in emotion_targets:
        normalized = annotations[target].astype("string").str.strip().str.lower()
        usable = normalized.isin({"yes", "no"}).to_numpy()
        model = fit_binary_logistic(
            base_features[usable],
            (normalized[usable] == "yes").astype(float).to_numpy(),
            iterations=int(config["binary_iterations"]),
            learning_rate=float(config["binary_learning_rate"]),
            l2=float(config["binary_l2"]),
        )
        emotion_models[target] = model
        emotion_probabilities.append(model.predict_probability(base_features))

    emotion_matrix = np.column_stack(emotion_probabilities)
    sentiment_features = np.column_stack([base_features, emotion_matrix])
    sentiment = annotations["sentiment"].astype("string").str.strip().str.lower()
    sentiment_classes = tuple(config["sentiment_classes"])
    usable_sentiment = sentiment.isin(sentiment_classes).to_numpy()
    sentiment_model = fit_softmax(
        sentiment_features[usable_sentiment],
        sentiment[usable_sentiment].tolist(),
        sentiment_classes,
        iterations=int(config["sentiment_iterations"]),
        learning_rate=float(config["sentiment_learning_rate"]),
        l2=float(config["sentiment_l2"]),
    )
    return ReactionFusionV2Model(
        version=str(config["version"]),
        base_feature_names=tuple(BASE_FEATURE_NAMES),
        emotion_targets=emotion_targets,
        emotion_models=emotion_models,
        sentiment_model=sentiment_model,
        temperature=1.0,
        abstention_threshold=float(config["minimum_abstention_threshold"]),
        smoothing_alpha=float(config["smoothing_alpha"]),
        metadata=dict(metadata or {}),
    )


def predict_v2_matrix(
    model: ReactionFusionV2Model, base_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    emotion_matrix = np.column_stack(
        [
            model.emotion_models[target].predict_probability(base_features)
            for target in model.emotion_targets
        ]
    )
    sentiment_features = np.column_stack([base_features, emotion_matrix])
    probabilities = model.sentiment_model.predict_probability(
        sentiment_features, model.temperature
    )
    return probabilities, emotion_matrix


def _decision_reason(
    audit: Mapping[str, float], emotion_probabilities: Mapping[str, float], candidate: str
) -> str:
    strongest = sorted(
        emotion_probabilities.items(), key=lambda item: item[1], reverse=True
    )[:2]
    emotion_text = ", ".join(f"{name}={value:.2f}" for name, value in strongest)
    return (
        f"candidate={candidate}; anchor_balance={audit['anchor_balance']:.3f}; "
        f"mixed_evidence={audit['mixed_evidence']:.3f}; strongest={emotion_text}"
    )


def fuse_reactions_v2(
    counts: Mapping[str, int | float], model: ReactionFusionV2Model
) -> ReactionFusionV2Result:
    base_features, audit = extract_reaction_features(counts, model.smoothing_alpha)
    sentiment_matrix, emotion_matrix = predict_v2_matrix(
        model, base_features.reshape(1, -1)
    )
    sentiment_probabilities = dict(
        zip(model.sentiment_model.classes, sentiment_matrix[0].tolist(), strict=True)
    )
    emotion_probabilities = dict(
        zip(model.emotion_targets, emotion_matrix[0].tolist(), strict=True)
    )
    candidate = max(sentiment_probabilities, key=sentiment_probabilities.get)
    confidence = float(sentiment_probabilities[candidate])
    is_uncertain = confidence < model.abstention_threshold
    return ReactionFusionV2Result(
        sentiment_label="uncertain" if is_uncertain else candidate,
        candidate_sentiment_label=candidate,
        label_confidence=confidence,
        is_uncertain=is_uncertain,
        sentiment_probabilities=sentiment_probabilities,
        emotion_probabilities=emotion_probabilities,
        reaction_entropy=audit["reaction_entropy"],
        mixed_evidence=audit["mixed_evidence"],
        decision_reason=_decision_reason(audit, emotion_probabilities, candidate),
    )
