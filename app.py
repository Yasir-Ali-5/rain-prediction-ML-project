import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib as j
import numpy as np
import pandas as pd

app = FastAPI()

# Root endpoint (GET)
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# GET method with query parameter
@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}

# Input schema using Pydantic
class Tomorrow(BaseModel):
    Location: str
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    WindGustSpeed: float
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float
    Cloud3pm: float
    Temp9am: float
    Temp3pm: float
    RainToday: str
    date: int
    month: int
    year: int

# POST method for prediction
@app.post('/userdata')
def receive_data(user: Tomorrow):
    load_model = j.load(r"models\model.pkl")
    encoder = j.load(r"models\encoder.pkl")

    # Convert user input to dictionary and then to DataFrame
    user_dict = user.dict()
    df = pd.DataFrame([user_dict])

    # Apply encoding
    df['Location'] = encoder['Location'].transform(df['Location'])
    df['RainToday'] = encoder['RainToday'].transform(df['RainToday'])

    # Prediction
    y_pred = load_model.predict(df)
    result = "Yes" if y_pred[0] == 1 else "No"

    return {
        "message": f"The Prediction for rain tomorrow is: {result}"
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
