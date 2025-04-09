import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Sample training data
data = pd.DataFrame({
    "area": [1000, 1500, 800, 1200, 950],
    "bedrooms": [2, 3, 1, 2, 2],
    "bathrooms": [2, 2, 1, 2, 1],
    "location": ["Dubai Marina", "Downtown Dubai", "Deira", "JLT", "Bur Dubai"],
    "price": [1400000, 2000000, 800000, 1500000, 1200000]
})

X = data[["location", "area", "bedrooms", "bathrooms"]]
y = data["price"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), ["location"])
    ],
    remainder="passthrough"
)

# Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train and save
pipeline.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Trained and saved new pipeline model.")

