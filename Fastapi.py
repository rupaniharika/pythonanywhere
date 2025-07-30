from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load saved model
model = joblib.load("iris_model.pkl")

# Initialize app
app = FastAPI(title="Iris Classifier API")

# Define request body
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define endpoint
@app.post("/predict")
def predict_species(data: IrisInput):
    try:
        input_array = np.array([[data.sepal_length, data.sepal_width,
                                 data.petal_length, data.petal_width]])
        prediction = model.predict(input_array)
        return {"predicted_class": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
