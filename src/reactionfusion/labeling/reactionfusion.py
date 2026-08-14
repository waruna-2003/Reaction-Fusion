"""Configurable ReactionFusion v1 label generation.

The initial weights are research hypotheses, not validated ground truth. They
must be frozen before evaluation and revised only against the training/validation
annotation subset, never the final held-out human test set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


REACTIONS = ("like", "love", "care", "haha", "wow", "sad", "angry")


@dataclass(frozen=True)
class FusionResult:
    sentiment_label: str
    label_confidence: float
    fusion_score: float
    reaction_entropy: float
    is_ambiguous: bool
    positive_anchor_mass: float
    negative_anchor_mass: float


def normalized_entropy(proportions: Mapping[str, float]) -> float:
    """Return Shannon entropy normalized to [0, 1] for seven reactions."""
    entropy = -sum(value * math.log(value) for value in proportions.values() if value > 0)
    return entropy / math.log(len(REACTIONS))


def fuse_reactions(
    counts: Mapping[str, int | float], config: Mapping[str, object]
) -> FusionResult:
    """Fuse all seven reactions into a provisional sentiment and confidence.

    Love/Care and Sad/Angry form the clear-valence anchor. Like, Haha, and Wow
    receive polarity from that anchor rather than being discarded or assigned a
    fixed meaning. When no clear-valence context exists, their net polarity is
    intentionally close to neutral while still influencing entropy/confidence.
    """
    values = {reaction: float(counts.get(reaction, 0)) for reaction in REACTIONS}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("ReactionFusion requires at least one reaction")

    proportions = {reaction: value / total for reaction, value in values.items()}
    entropy = normalized_entropy(proportions)
    anchor_weights = config["anchor_weights"]
    ambiguous_weights = config["ambiguous_context_weights"]
    assert isinstance(anchor_weights, Mapping)
    assert isinstance(ambiguous_weights, Mapping)

    anchor = sum(
        proportions[reaction] * float(weight)
        for reaction, weight in anchor_weights.items()
    )
    context = math.tanh(float(config["anchor_context_scale"]) * anchor)
    ambiguous_mass = sum(
        proportions[reaction] * float(weight)
        for reaction, weight in ambiguous_weights.items()
    )
    score = max(-1.0, min(1.0, anchor + context * ambiguous_mass))

    positive_mass = proportions["love"] + 0.55 * proportions["care"]
    negative_mass = proportions["sad"] + proportions["angry"]
    opposition = min(positive_mass, negative_mass)
    positive_threshold = float(config["positive_threshold"])
    negative_threshold = float(config["negative_threshold"])

    if score >= positive_threshold:
        label = "positive"
    elif score <= negative_threshold:
        label = "negative"
    elif (
        entropy >= float(config["mixed_entropy_threshold"])
        and opposition >= float(config["mixed_opposition_threshold"])
    ):
        label = "mixed"
    else:
        label = "neutral"

    engagement = 1.0 - math.exp(-total / float(config["engagement_scale"]))
    if label in {"positive", "negative"}:
        confidence = abs(score) * (0.5 + 0.5 * (1.0 - entropy)) * engagement
    elif label == "mixed":
        opposition_factor = min(
            1.0, opposition / max(float(config["mixed_opposition_threshold"]), 1e-9)
        )
        confidence = entropy * opposition_factor * engagement
    else:
        distance_factor = 1.0 - min(
            1.0, abs(score) / max(abs(positive_threshold), abs(negative_threshold))
        )
        confidence = distance_factor * (0.5 + 0.5 * entropy) * engagement

    confidence = max(0.0, min(1.0, confidence))
    ambiguous = label in {"neutral", "mixed"} or confidence < float(
        config["low_confidence_threshold"]
    )
    return FusionResult(
        sentiment_label=label,
        label_confidence=confidence,
        fusion_score=score,
        reaction_entropy=entropy,
        is_ambiguous=ambiguous,
        positive_anchor_mass=positive_mass,
        negative_anchor_mass=negative_mass,
    )


def filtered_baseline(counts: Mapping[str, int | float], threshold: float = 0.05) -> tuple[str, float]:
    """Baseline that discards Like/Haha/Wow and compares clear reactions."""
    positive = float(counts.get("love", 0)) + 0.55 * float(counts.get("care", 0))
    negative = float(counts.get("sad", 0)) + float(counts.get("angry", 0))
    denominator = positive + negative
    score = 0.0 if denominator == 0 else (positive - negative) / denominator
    if score >= threshold:
        return "positive", score
    if score <= -threshold:
        return "negative", score
    return "neutral", score
