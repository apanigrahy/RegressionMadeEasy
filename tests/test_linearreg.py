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
OUTCOME = "TWA MAP"
PREDICTORS = ["Sleeptime"]

# Convert to numpy for statsmodels
y = df[OUTCOME].to_numpy()
X = df[PREDICTORS].to_numpy()

# Add an intercepts column
X = sm.add_constant(X)

# Fit an Ordinary Least Squares (OLS) linear regression model
model = sm.OLS(y, X).fit()

# Print summary to check if model was fit
print(model.summary())