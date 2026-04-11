"""Tests for LogisticMadeEasy class (from logisticreg.py)."""

# Import libraries
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from plotnine import ggplot

# Import LogisticMadeEasy class object
from logisticreg import LogisticMadeEasy


# Simulate sample data, fit logistic regression model, and return a
# LogisticMadeEasy object for testing
# Use pytest.fixture to fix the model fit once and reuse
@pytest.fixture
def logistic_model() -> LogisticMadeEasy:
    """Simulate data and fit logistic regression model for testing."""
    # Seet seed
    rng = np.random.default_rng(42)
    # Set sample size
    n = 100
    # Simulate predictor variables
    age = rng.normal(55, 10, n)
    bmi = rng.normal(27, 5, n)
    # Calculate logit probabilities
    log_odds = -8 + 0.08 * age + 0.1 * bmi
    prob = 1 / (1 + np.exp(-log_odds))
    # Simulate from binomial distribution binary outcome variables
    diabetes = rng.binomial(1, prob, n)
    # Create pandas data frame
    df = pd.DataFrame({"age": age, "bmi": bmi, "diabetes": diabetes})
    X = sm.add_constant(df[["age", "bmi"]])
    # Fit the logistic regression model
    model = sm.Logit(df["diabetes"], X).fit(disp=False)
    # Return a LogisticMadeEasy model
    return LogisticMadeEasy(model)


# Test that residual vs fitted plot property returns a ggplot object
def test_deviance_residual_vs_fitted_returns_ggplot(
    logistic_model: LogisticMadeEasy,
) -> None:
    """Test that deviance residual vs fitted plot property returns a ggplot."""
    plot = logistic_model.deviance_residual_vs_fitted_plot()
    assert isinstance(plot, ggplot)


# Test that cook's distance plot property returns a ggplot object
def test_cooks_distance_returns_ggplot(
    logistic_model: LogisticMadeEasy,
) -> None:
    """Test that cook's distance plot property returns a ggplot."""
    plot = logistic_model.cooks_distance_plot()
    assert isinstance(plot, ggplot)


# Test that dfbetas plot property returns a ggplot object
def test_dfbetas_returns_ggplot(
    logistic_model: LogisticMadeEasy,
) -> None:
    """Test that dfbetas plot property returns a ggplot."""
    plot = logistic_model.dfbetas_plot()
    assert isinstance(plot, ggplot)


# Test that VIF plot property returns a ggplot object
def test_vif_returns_ggplot(
    logistic_model: LogisticMadeEasy,
) -> None:
    """Test that VIF plot property returns a ggplot."""
    plot = logistic_model.vif_plot()
    assert isinstance(plot, ggplot)


# Test that ROC curve plot property returns a ggplot object
def test_roc_curve_returns_ggplot(
    logistic_model: LogisticMadeEasy,
) -> None:
    """Test that ROC curve plot property returns a ggplot."""
    plot = logistic_model.roc_curve_plot()
    assert isinstance(plot, ggplot)

# Test that an error is raised if a non-logistic model is passed into function
def test_invalid_model_raises_type_error() -> None:
    """Test that a TypeError is raised for a non-logistic model."""
    with pytest.raises(TypeError):
        LogisticMadeEasy("not a model")  
