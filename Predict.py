import joblib as j
import numpy as np
import pandas as pd

load_model = j.load(r"models\model.pkl")
encoder = j.load(r"models\encoder.pkl")

columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'date', 'month', 'year']

user_inputs = {}
for i in columns:
    value = input(f"Enter {i}: ")
    user_inputs[i] = [value]

user_inputs['Location'] = encoder['Location'].transform(user_inputs['Location'])
user_inputs['RainToday'] = encoder['RainToday'].transform(user_inputs['RainToday'])

user_inputs = pd.DataFrame(user_inputs)

print(user_inputs)
y_pred = load_model.predict(user_inputs)
print(f"Prediction: {y_pred[0]}")