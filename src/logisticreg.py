"""Create plots for logistic regression models."""

# Import pathlib for handling files
import pathlib
from typing import cast

# Import packages for data manipulation and regression model fitting
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm
from plotnine import (
    aes,
    facet_wrap,
    geom_hline,
    geom_point,
    geom_segment,
    ggplot,
    labs,
    theme_classic,
)


# Create a custom class object for logistic regression models
class LogisticMadeEasy:
    """Diagnostic plots for logistic regression models."""

    # Define instances
    def __init__(self, model: dm.BinaryResultsWrapper) -> None:
        """Intialize instances for LogisticMadeEasy class."""
        self.model = model
        self.dfbetas = model.get_influence().dfbetas
        self.predictor_names = model.model.exog_names
        self.fitted_values = model.fittedvalues
        self.observation_number = np.arange(len(self.fitted_values))
        self.deviance_residuals = model.resid_dev
        self.cooks_distance = model.get_influence().cooks_distance[0]

    # Define a property for fixing the predictor names

    # Define a property for a deviance residual vs fitted plot
    def deviance_residual_vs_fitted_plot(self) -> ggplot:
        """Plot deviance residuals vs fitted values."""
        # Build a dataframe for plotnine to use
        plot_df = pd.DataFrame(
            {
                "fitted_values": self.fitted_values,
                "deviance_residuals": self.deviance_residuals,
            }
        )

        # Return the plot
        return (
            ggplot(plot_df, 
            aes( # type: ignore[no-untyped-call]
                x="fitted_values", y="deviance_residuals"))  
            + geom_hline(yintercept=0, linetype="dashed", color="red")
            + geom_point(alpha=0.5, color="skyblue")
            + theme_classic()  # type: ignore[no-untyped-call]
            + labs(
                title="Deviance Residuals vs Fitted Values",
                x="Fitted Values",
                y="Deviance Residuals",
            )
        )

    # Define a property for cook's distance plot
    def cooks_distance_plot(self) -> ggplot:
        """Plot the cook's distance for model."""
        # Build a dataframe for plotnine to use
        plot_df = pd.DataFrame(
            {
                "observation_number": self.observation_number,
                "cooks_distance": self.cooks_distance,
            }
        )

        # Return the plot
        return cast(
            ggplot,
            ggplot(
                plot_df,
                aes(  # type: ignore[no-untyped-call]
                    x="observation_number", y="cooks_distance"
                ),
            )
            + geom_point(alpha=0.8, color="skyblue")
            + geom_segment(
                aes(  # type: ignore[no-untyped-call]
                    x="observation_number",
                    xend="observation_number",
                    y=0,
                    yend="cooks_distance",
                ),
                alpha=0.5,
                color="grey",
            )
            + geom_hline(
                yintercept=(4 / len(plot_df)), linetype="dashed", color="red"
            )
            + theme_classic()  # type: ignore[no-untyped-call]
            + labs(
                title="Cook's Distance vs Observation Number",
                x="Observation Number",
                y="Cook's Distance",
            ),  # type: ignore[operator]
        )

    # Define a property for dfbetas plot
    def dfbetas_plot(self) -> ggplot:
        """Plot dfbetas for each predictor."""
        # Change "const" to be "Intercept"
        self.predictor_names = [
            "Intercept" if name == "const" else name
            for name in model.model.exog_names
        ]
        # Get number of observations
        n = len(self.observation_number)
        # Calculate theortical threshold
        threshold = 2 / np.sqrt(n)
        # Create a pandas dataframe for plotting (including intercept)
        plot_df = pd.DataFrame(
            self.dfbetas,
            columns=self.predictor_names,
        )
        # Get the observation number
        plot_df["observation_number"] = self.observation_number

        # Convert to a long dataset
        plot_df = plot_df.melt(
            id_vars="observation_number",
            var_name="predictor",
            value_name="dfbeta",
        )

        # Return the plot
        return cast(
            ggplot,
            ggplot(
                plot_df,
                aes(  # type: ignore[no-untyped-call]
                    x="observation_number", y="dfbeta"
                ),
            )
            + geom_point(alpha=0.8, color="skyblue")
            + geom_segment(
                aes(  # type: ignore[no-untyped-call]
                    x="observation_number",
                    xend="observation_number",
                    y=0,
                    yend="dfbeta",
                ),
                alpha=0.5,
                color="grey",
            )
            + geom_hline(
                yintercept=threshold,
                linetype="dashed",
                color="red",
            )
            + geom_hline(
                yintercept=-threshold,
                linetype="dashed",
                color="red",
            )
            + facet_wrap("predictor")  
            + theme_classic()  # type: ignore[no-untyped-call]
            + labs(
                title="DFBetas by Predictor",
                x="Observation Number",
                y="DFBeta",
            ),  # type: ignore[operator]
        )


    # Define a property for VIF plot

    # Define a method for displaying a coefficent summary table

    # Define a method for displaying regression model summary table


# Testing
if __name__ == "__main__":
    # Set path to sample data
    DATA_PATH = (
        pathlib.Path(__file__).parent.parent
        / "tests"
        / "data"
        / "diabetes_sample_data_BRFSS2015.csv"
    )

    # Read in the data
    df = pd.read_csv(DATA_PATH)

    # Define outcome and predictor variables
    OUTCOME = "Diabetes_binary"
    PREDICTORS = ["HighBP", "HighChol", "BMI", "Smoker"]

    # Convert to numpy for statsmodels
    y = df[OUTCOME].to_numpy()
    X = df[PREDICTORS].to_numpy()

    # Add an intercepts column
    X = sm.add_constant(X)

    # Fit a logistic regression model
    model = sm.Logit(y, X).fit()

    # Print summary to check if model was fit
    print(model.summary())

    # Print model type
    print(type(model))

    # Create the made easy object
    logistic_diag = LogisticMadeEasy(model)

    # Create residual vs fitted plot
    logistic_diag.deviance_residual_vs_fitted_plot().save(
        "deviance_residual_vs_fitted.png"
    )

    # Create cook's distance vs oobservation number plot
    logistic_diag.cooks_distance_plot().save(
        "cooks_distance_vs_obs_number.png"
    )

    # Create a dfbetas plot
    logistic_diag.dfbetas_plot().save(
        "dfbetas_plot.png"
    )
