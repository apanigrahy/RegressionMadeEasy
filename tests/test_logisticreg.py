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