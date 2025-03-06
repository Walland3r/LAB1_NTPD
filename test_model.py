import joblib
import numpy as np

# Load the saved model
model = joblib.load('model.joblib')

# Example record for prediction
example_record = np.random.uniform(0.1, 8, (1, 4))

# Perform prediction
prediction = model.predict(example_record)

print(f'Values: {example_record}')
print(f'Prediction: {prediction}')