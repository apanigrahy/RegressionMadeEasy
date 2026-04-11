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
    / "diabetes_sample_data_BRFSS2015.csv"
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scipy.stats as stats
from typing import Any
from statsmodels.regression.linear_model import RegressionResultsWrapper
from plotnine import ggplot, aes, geom_point, geom_smooth, labs, theme_minimal, geom_hline, stat_qq, stat_qq_line


class LinearMadeEasy:
    """Helper class for plots, tables, & interpretations from a lin model."""

    def __init__(
        self,
        model: RegressionResultsWrapper,
    ) -> None:
        """Initialize the LinearMadeEasy object.

        This only requires the fitted model results object. 
        Everything else cna be extracted since the X and Y data 
        is tucked inside of the model object (exog and endog).
        """
        self.model = model
        self.x = model.model.exog
        self.y = model.model.endog
        self.predictor_x = model.model.exog_names
        self.predictor_y = model.model.endog_names


    @property
    def diagnostic_data(self) -> pd.DataFrame:
        """Internal helper to bundle residuals and fitted values into a DataFrame."""
        return pd.DataFrame({
            "fitted": self.model.fittedvalues,
            "residuals": self.model.resid,
            "std_resid": self.model.get_influence().resid_studentized_internal
        })

    @property
    def resid_vs_fitted(self) -> ggplot:
        """Creates a residuals vs fitted scatter plot."""
        df = self.diagnostic_data
        
        return (
            ggplot(df, aes(x='fitted', y='residuals'))
            + geom_point(alpha=0.3, color="skyblue")
            + geom_hline(yintercept=0, color="red", linetype="dashed")
            + labs(title="Residuals vs Fitted", x="Predicted", y="Error")
            + theme_minimal()
        )

    @property
    def qq_plot(self) -> ggplot:
        """Creates a Normal Q-Q plot to check if residuals are normally distributed."""
        df = self.diagnostic_data
        return (
            ggplot(df, aes(sample='std_resid')) 
            + stat_qq()
            + stat_qq_line(color="red", linetype="dashed")
            + labs(
                title="Normal Q-Q Plot")
            + theme_minimal()
        )
        

    @property
    def plot_regression(self) -> ggplot:
        """Create a scatter plot with fitted regression line.

        Returns:
            A Figure object showing observed data and fitted line.
        """
        df_plot = pd.DataFrame({
            'x_val': self.x[:, 1],
            'y_val': self.y
        })

        return (
        ggplot(df_plot, aes(x='x_val', y='y_val'))
        + geom_point(alpha=0.1, color="skyblue")
        + geom_smooth(method='lm', color='red', size=1.5) 
        + labs(
            title=f"Linear Regression: {self.predictor_x[1]} vs {self.predictor_y}",
            x=self.predictor_x[1],
            y=self.predictor_y,
            caption="Line calculated via OLS"
        )
        + theme_minimal()
    )

helper = LinearMadeEasy(model)

import matplotlib.pyplot as plt

# 2. Test the Regression Plot
print("Generating Regression Plot...")
reg_plot = helper.plot_regression
#reg_plot.save("regression_analysis.png", width=8, height=6, dpi=300)

#3 Test Resid 
print("Generating Residual Plot...")
res_plot = helper.resid_vs_fitted
res_plot.save("resid_plot.png", width=8, height=6, dpi=300)

# 4. Test the QQ Plot
print("Generating QQ Plot...")
qq_plot = helper.qq_plot
qq_plot.save("qq_plot.png", width=8, height=6, dpi=300)