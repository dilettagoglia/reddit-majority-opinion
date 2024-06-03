#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

# Sample data
data = {
    'YTA_pre': [0.6, 0.4, 0.5],
    'NTA_pre': [0.3, 0.4, 0.2],
    'ESH_pre': [0.1, 0.1, 0.2],
    'NAH_pre': [0.0, 0.1, 0.1],
    'YTA_post': [0.7, 0.35, 0.6],
    'NTA_post': [0.2, 0.45, 0.15],
    'ESH_post': [0.1, 0.15, 0.2],
    'NAH_post': [0.0, 0.05, 0.05],
    'Verdict': ['YTA', 'NTA', 'YTA']
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate differences
df['YTA_diff'] = df['YTA_post'] - df['YTA_pre']
df['NTA_diff'] = df['NTA_post'] - df['NTA_pre']
df['ESH_diff'] = df['ESH_post'] - df['ESH_pre']
df['NAH_diff'] = df['NAH_post'] - df['NAH_pre']

# One-hot encode the verdict
encoder = OneHotEncoder(drop='first')
verdict_encoded = encoder.fit_transform(df[['Verdict']]).toarray()
verdict_labels = encoder.categories_[0][1:]

# Prepare feature matrix X and target vectors y
X = np.hstack([df[['YTA_pre', 'NTA_pre', 'ESH_pre', 'NAH_pre']], verdict_encoded])
y = df[['YTA_diff', 'NTA_diff', 'ESH_diff', 'NAH_diff']]

# Add a constant to the feature matrix for the intercept
X = sm.add_constant(X)

# Fit Bayesian regression model for each target variable
models = []
for i in range(y.shape[1]):
    y_i = y.iloc[:, i]
    model = sm.GLM(y_i, X, family=sm.families.Gaussian())
    result = model.fit()
    models.append(result)

# Print summary for each model
for i, result in enumerate(models):
    print(f"Model for {y.columns[i]}")
    print(result.summary())
    print("\n")


# %%
