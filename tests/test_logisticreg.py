"""Tests for logisticreg.py."""

# Import pathlib for handling files
import pathlib

# Import packages for data manipulation and regression model fitting
import pandas as pd
import statsmodels.api as sm

# Set path to sample data
DATA_PATH = (
    pathlib.Path(__file__).parent
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
