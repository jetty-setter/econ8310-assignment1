import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------
# Load data
# -----------------------------
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

y_train = train["trips"]

# -----------------------------
# Define model
# -----------------------------
model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal="mul",       # 🔥 KEY CHANGE
    seasonal_periods=168
)

# -----------------------------
# Fit model
# -----------------------------
modelFit = model.fit(optimized=True)

# -----------------------------
# Forecast January (744 hours)
# -----------------------------
pred = modelFit.forecast(744)
pred = np.array(pred)