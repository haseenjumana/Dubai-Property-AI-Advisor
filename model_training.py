import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ✅ Step 1: Sample data
data = pd.DataFrame({
    "area": [1000, 1500, 800, 1200, 950],
    "bedrooms": [2, 3, 1, 2, 2],
    "bathrooms": [2, 2, 1, 2, 1],
    "location": ["Dubai Marina", "Downtown Dubai", "Deira", "JLT", "Bur Dubai"],
    "price": [1400000, 2000000, 800000, 1500000, 1200000]
})

# ✅ Step 2: Split features/target
X = data.drop("price", axis=1)
y = data["price"]

# ✅ Step 3: Preprocessing for categorical data
categorical_features = ["location"]
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[("cat", categorical_transformer, categorical_features)],
    remainder="passthrough"  # Keep other columns (area, bedrooms, bathrooms)
)

# ✅ Step 4: Create pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# ✅ Step 5: Train
model.fit(X, y)

# ✅ Step 6: Save full pipeline
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")

