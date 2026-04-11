# RegressionMadeEasy

Linear & Logistic Regression Diagnostics Library

Overview:
We have created a Python library designed to simplify regression diagnostics for both linear and logistic regression models. 
The goal of this project is to take a fitted model object (from statsmodels) and automatically generate standard diagnostic plots used to evaluate model assumptions, detect outliers, and assess overall model quality.
Instead of manually extracting residuals, fitted values, and influence measures, this library provides ready-to-use visualizations that are directly tied to your model and data.

Input:
Both classes (LinearMadeEasy & LogisticMadeEasy) in this library require a fitted regression model from statsmodel.

The core idea is that each class will accept the fitted model object, extract relevant diagnostic quantities (residuals, fitted values, influence, etc), store them internally, and provide plotting methods as properties for easy access. 

For LinearMadeEasy, there are 5 properties:

diagnostic_data - Creates a structured dataset containing key diagnostic values and returns fitted values, residuals, and standardized residuals. This serves as the foundation for the other plots which ensures consistency and avoides repeated computation.

resid_vs_fitted - Plots residuals against fitted values testing for linearity asusmption and homoscadasticity. Random scatter is good, patterns or curves can be an indication for possible issues.

qq_plot - Creates a normal QQ-Plot of standardized residuals. This checks the normaility of residuals. Points on the line indicates a normal distribution, and deviations from the line like heavy tails can indicate possible skewness.

regression_plot - Visualizes observed data along with the fitted regression line. This shows the relationship between predictor and response, and how well the model fits the data.

cooks_distance_plot - Identifies influential observations using Cook’s Distance by checking the influence of each data point on the fitted model. Points above the threshhold may strongly influence the model and should be checked.

For LogitMadeEasy there are :





Example Data Set:
hypoxia.csv


