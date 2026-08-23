import math

import pandas as pd

from reactionfusion.evaluation.human_validation import (
    classification_metrics,
    cohen_kappa,
    exact_mcnemar,
)


def test_cohen_kappa_perfect_agreement() -> None:
    labels = pd.Series(["positive", "negative", "neutral", "positive"])
    assert cohen_kappa(labels, labels) == 1.0


def test_classification_metrics_include_unpredicted_mixed_class() -> None:
    truth = pd.Series(["negative", "neutral", "positive", "mixed", "uncertain"])
    prediction = pd.Series(["negative", "neutral", "positive", "neutral", "positive"])
    metrics = classification_metrics(truth, prediction)
    assert metrics["evaluated_records"] == 4
    assert metrics["excluded_uncertain_records"] == 1
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["mixed"]["recall"] == 0.0


def test_exact_mcnemar_is_symmetric() -> None:
    truth = pd.Series(["positive", "positive", "negative", "negative"])
    first = pd.Series(["positive", "negative", "negative", "positive"])
    second = pd.Series(["negative", "positive", "positive", "negative"])
    result = exact_mcnemar(truth, first, second)
    assert result["discordant_records"] == 4
    assert math.isclose(result["two_sided_exact_p_value"], 1.0)
