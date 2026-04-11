"""Utilities for interpreting linear regression models."""

# Import pathlib for handling files
import pathlib


import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import cast
from statsmodels.regression.linear_model import RegressionResultsWrapper
from plotnine import (
    ggplot, aes, geom_point, geom_hline, geom_smooth,
    labs, theme_classic, stat_qq, stat_qq_line
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
            ggplot(df, aes(x="fitted", y="residuals")) # type: ignore[no-untyped-call]
            + geom_point(alpha=0.4, color = "skyblue")
            + geom_hline(yintercept=0, linetype="dashed", color="red")
            + labs(
                title="Residuals vs Fitted Values",
                x="Fitted Values",
                y="Residuals"
            )
            + theme_classic() # type: ignore[no-untyped-call]
        )


    @property
    def qq_plot(self) -> ggplot:
        """QQ plot for normality of residuals."""
        df = self.diagnostic_data

        return (
            ggplot(df, aes(sample="std_resid")) # type: ignore[no-untyped-call]
            + stat_qq()
            + stat_qq_line(color="red", linetype="dashed")
            + labs(title="Normal Q-Q Plot (Standardized Residuals)",
                   x="Theoretical Quantiles",
                   y="Sample Quantiles"
            )
            + theme_classic() # type: ignore[no-untyped-call]
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
            ggplot(df_plot, aes(x="x", y="y")) # type: ignore[no-untyped-call]
            + geom_point(alpha=0.3, color = "skyblue")
            + geom_smooth(method="lm")
            + labs(
                title=f"{self.response_name} vs Predictor",
                x=self.predictor_names[1] if len(self.predictor_names) > 1 else self.predictor_names[0],
                y=self.response_name
            )
            + theme_classic() # type: ignore[no-untyped-call]
        )
        

# Testing
if __name__ == "__main__":
    # Set path to sample data
    DATA_PATH = (
        pathlib.Path(__file__).parent.parent
        / "tests"
        / "data"
        / "hypoxia.csv"
    )

    # Read in the data
    df = pd.read_csv(DATA_PATH)

    # Define outcome and predictor variables
    OUTCOME = "TWA MAP"
    PREDICTORS = ["Sleeptime"]

    # Define outcome and predictor variables
    y = df[OUTCOME]
    X = sm.add_constant(df[PREDICTORS])

    # Fit a linear regression model
    model = sm.OLS(y, X).fit()

    # Print summary to check if model was fit
    print(model.summary())

    # Print model type
    print(type(model))

    # Create the made easy object
    helper = LinearMadeEasy(model)

    #1 Test Regression Plot
    reg_plot = helper.regression_plot
    reg_plot.save("reg_analysis3.png", width=8, height=6, dpi=300)

    #2 Test Resid Plot
    res_plot = helper.resid_vs_fitted
    res_plot.save("resid_plot3.png", width=8, height=6, dpi=300)

    #3 Test the QQ Plot
    qq_plot = helper.qq_plot
    qq_plot.save("qq_plot3.png", width=8, height=6, dpi=300)

