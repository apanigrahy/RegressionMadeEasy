"""Tests for logisticreg.py."""

# Set path to sample data
DATA_PATH = (
    pathlib.Path(__file__).parent.parent / "tests" / "data" / "hypoxia.csv"
)

# Read in the data
df = pd.read_csv(DATA_PATH)

# Define outcome and predictor variables
OUTCOME = "CAD"
PREDICTORS = ["Age", "BMI", "Hyper", "Sleeptime"]

# Define outcome and predictor variables
y = df[OUTCOME]
X = sm.add_constant(df[PREDICTORS])

# Fit a logistic regression model
model = sm.Logit(y, X).fit()
