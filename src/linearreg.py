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
        
"""
    @property
    def table_one(self) -> pd.DataFrame:
        Generate a summary table of regression coefficients.

        Returns:
            A pandas DataFrame containing coefficients, standard errors,
            p-values, confidence intervals, and significance labels.
        
        conf_int = self.model.conf_int()

        df_results = pd.DataFrame({
            "Variable": self.predictor_names,
            "Coefficient": self.model.params,
            "Std_Error": self.model.bse,
            "P_Value": self.model.pvalues,
            "Conf_Lower": conf_int[:, 0],
            "Conf_Upper": conf_int[:, 1],
        })

        df_results["Significance"] = df_results["P_Value"].apply(
            lambda p: "Significant" if p < 0.05 else "Not Significant"
        )

        return df_results

    @property
    def interpretation(self) -> str:
        Generate a plain-English interpretation of the regression result.

        Returns:
            A string summarizing the direction, magnitude, and statistical
            significance of the predictor.

        Notes:
            - Assumes a single predictor model.
            - Interprets the coefficient of the first non-intercept term.
        
        coef = float(self.model.params[1])
        pval = float(self.model.pvalues[1])

        significance = (
            "statistically significant" if pval < 0.05 else "not statistically significant"
        )

        direction = "increase" if coef > 0 else "decrease"

        return (
            f"The model found a {significance} relationship between "
            f"{self.predictor_names[1]} and the response. "
            f"For every 1-unit increase in {self.predictor_names[1]}, "
            f"the response is expected to {direction} by {abs(coef):.4f} units."
        )
    
"""
