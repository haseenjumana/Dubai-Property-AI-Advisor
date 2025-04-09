import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Sample dataset
data = pd.DataFrame({
    "area": [1000, 1500, 2000, 1200, 1800],
    "bedrooms": [2, 3, 4, 2, 3],
    "bathrooms": [2, 2, 3, 2, 3],
    "location": ["Dubai Marina", "Downtown Dubai", "Deira", "Bur Dubai", "JLT"],
    "price": [1400000, 2000000, 1600000, 1500000, 1800000]
})

X = data[["area", "bedrooms", "bathrooms", "location"]]
y = data["price"]

# Preprocessing for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), ["location"])
    ],
    remainder='passthrough'
)

# Create pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train model
model.fit(X, y)

# Save model using joblib
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")


