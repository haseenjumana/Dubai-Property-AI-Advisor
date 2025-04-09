import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load data
df = pd.read_csv("data.csv")

X = df.drop("price", axis=1)
y = df["price"]

# Preprocessing: One-hot encode location
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), ["location"])
], remainder="passthrough")

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
