"""Unit tests for synthetic external-test metrics."""

import numpy as np
import pandas as pd

from reactionfusion.evaluation.synthetic_validation import (
    paired_exact_mcnemar,
    probability_metrics,
)


def test_probability_metrics_are_zero_for_perfect_predictions() -> None:
    truth = pd.Series(["negative", "neutral", "positive", "mixed"])
    probabilities = np.eye(4)
    metrics = probability_metrics(truth, probabilities)
    assert metrics["multiclass_log_loss"] < 1e-9
    assert metrics["multiclass_brier_score"] == 0.0
    assert metrics["expected_calibration_error_10_bins"] == 0.0


def test_paired_test_counts_discordant_predictions() -> None:
    truth = pd.Series(["positive", "positive", "negative", "negative"])
    first = pd.Series(["positive", "positive", "positive", "negative"])
    second = pd.Series(["negative", "positive", "negative", "negative"])
    result = paired_exact_mcnemar(truth, first, second)
    assert result["first_correct_second_wrong"] == 1
    assert result["first_wrong_second_correct"] == 1
    assert result["discordant_records"] == 2
