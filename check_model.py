import joblib

# Load the model
model = joblib.load('model.pkl')

# Print out the model pipeline
print("✅ Model loaded successfully!")
print("Model type:", type(model))
print("\nModel structure:\n")
print(model)
