"""Tests for linearreg.py."""

# Import pathlib for handling files
import pathlib

# Import packages for data manipulation and regression model fitting
import polars as pl
import statsmodels.api as sm

# Set path to sample data
DATA_PATH = (
    pathlib.Path(__file__).parent
    / "data"
    / "hypoxia.csv"
)

# Read in the data
df = pl.read_csv(DATA_PATH)

# Define outcome and predictor variables
OUTCOME = "Income"
PREDICTORS = ["Age"]

# Convert to numpy for statsmodels
y = df[OUTCOME].to_numpy()
X = df[PREDICTORS].to_numpy()

# Add an intercepts column
X = sm.add_constant(X)

# Fit an Ordinary Least Squares (OLS) linear regression model
model = sm.OLS(y, X).fit()

# Print summary to check if model was fit
#print(model.summary())


"""Utilities for interpreting linear regression models."""

"""Utilities for interpreting linear regression models."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import cast
from statsmodels.regression.linear_model import RegressionResultsWrapper
from plotnine import (
    ggplot, aes, geom_point, geom_hline, geom_smooth,
    labs, theme_minimal, stat_qq, stat_qq_line
)


class LinearMadeEasy:
    """Helper class for diagnostic plots for linear regression models."""

    def __init__(self, model: RegressionResultsWrapper) -> None:
        """Initialize with fitted OLS model."""
        self.model = model

        self.fitted_values = model.fittedvalues
        self.residuals = model.resid
        self.std_resid = model.get_influence().resid_studentized_internal

        self.predictor_names = model.model.exog_names
        self.response_name = model.model.endog_names

        # safer design: store full design matrix
        self.X = model.model.exog
        self.y = model.model.endog


    @property
    def diagnostic_data(self) -> pd.DataFrame:
        """Bundle diagnostics into a DataFrame."""
        return pd.DataFrame({
            "fitted": self.fitted_values,
            "residuals": self.residuals,
            "std_resid": self.std_resid
        })


    @property
    def resid_vs_fitted(self) -> ggplot:
        """Residuals vs fitted values plot."""
        df = self.diagnostic_data

        return (
            ggplot(df, aes(x="fitted", y="residuals"))
            + geom_point(alpha=0.4)
            + geom_hline(yintercept=0, linetype="dashed", color="red")
            + labs(
                title="Residuals vs Fitted Values",
                x="Fitted Values",
                y="Residuals"
            )
            + theme_minimal()
        )


    @property
    def qq_plot(self) -> ggplot:
        """QQ plot for normality of residuals."""
        df = self.diagnostic_data

        return (
            ggplot(df, aes(sample="std_resid"))
            + stat_qq()
            + stat_qq_line(color="red", linetype="dashed")
            + labs(title="Normal Q-Q Plot (Standardized Residuals)")
            + theme_minimal()
        )


    @property
    def regression_plot(self) -> ggplot:
        """Observed data + fitted regression line."""
        x_vals = self.X[:, 1] if self.X.shape[1] > 1 else self.X[:, 0]

        df_plot = pd.DataFrame({
            "x": x_vals,
            "y": self.y
        })

        return (
            ggplot(df_plot, aes(x="x", y="y"))
            + geom_point(alpha=0.3)
            + geom_smooth(method="lm")
            + labs(
                title=f"{self.response_name} vs Predictor",
                x=self.predictor_names[1] if len(self.predictor_names) > 1 else self.predictor_names[0],
                y=self.response_name
            )
            + theme_minimal()
        )

helper = LinearMadeEasy(model)

import matplotlib.pyplot as plt

# 2. Test the Regression Plot
print("Generating Regression Plot...")
reg_plot = helper.regression_plot
reg_plot.save("regression_analysis.png", width=8, height=6, dpi=300)

#3 Test Resid 
print("Generating Residual Plot...")
res_plot = helper.resid_vs_fitted
res_plot.save("resid_plot.png", width=8, height=6, dpi=300)

# 4. Test the QQ Plot
print("Generating QQ Plot...")
qq_plot = helper.qq_plot
qq_plot.save("qq_plot.png", width=8, height=6, dpi=300)