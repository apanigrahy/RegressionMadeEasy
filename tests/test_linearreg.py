"""Tests for LinearMadeEasy class (from linearreg.py)."""

# Import libraries
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from plotnine import ggplot

# Import LogisticMadeEasy class object
from linearreg import LinearMadeEasy


# Simulate sample data, fit linear regression model, and return a
# LinearMadeEasy object for testing
# Use pytest.fixture to fix the model fit once and reuse
@pytest.fixture
def linear_model() -> LinearMadeEasy:
    """Simulate data and fit linear regression model for testing."""
    rng = np.random.default_rng(42)
    n = 100

    # Simulate predictor(s)
    x = rng.normal(10, 2, n)

    # Simulate response with noise
    y = 3 + 2 * x + rng.normal(0, 2, n)

    # Create DataFrame
    df = pd.DataFrame({"x": x, "y": y})

    # Fit OLS model
    X = sm.add_constant(df[["x"]])
    model = sm.OLS(df["y"], X).fit()

    return LinearMadeEasy(model)


def test_resid_vs_fitted(
    linear_model: LinearMadeEasy,
) -> None:
    """Test residual vs fitted plot returns a ggplot."""
    plot = linear_model.resid_vs_fitted
    assert isinstance(plot, ggplot)


def test_qq_plot(
    linear_model: LinearMadeEasy,
) -> None:
    """Test QQ plot returns a ggplot."""
    plot = linear_model.qq_plot
    assert isinstance(plot, ggplot)


def test_regression_plot(
    linear_model: LinearMadeEasy,
) -> None:
    """Test regression plot returns a ggplot."""
    plot = linear_model.regression_plot
    assert isinstance(plot, ggplot)


def test_cooks_distance(
    linear_model: LinearMadeEasy,
) -> None:
    """Test Cook's distance plot returns a ggplot."""
    plot = linear_model.cooks_distance_plot
    assert isinstance(plot, ggplot)


# Tests data structure 
def test_diagnostic_data(
    linear_model: LinearMadeEasy,
) -> None:
    """Test diagnostic_data contains expected columns."""
    df = linear_model.diagnostic_data

    assert "fitted" in df.columns
    assert "residuals" in df.columns
    assert "std_resid" in df.columns

# Test for invalid model error
def test_invalid_model_raises_type_error() -> None:
    """Test that a TypeError is raised for invalid model input."""
    with pytest.raises(TypeError):
        LinearMadeEasy("not a model")