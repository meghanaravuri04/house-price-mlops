import logging
import joblib
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="House Price Prediction API",
    version="1.0"
)

MODEL_PATH = "models/model.pkl"


class HouseData(BaseModel):

    area: float = Field(gt=0)
    bedrooms: int = Field(gt=0)
    bathrooms: int = Field(gt=0)
    house_age: float = Field(ge=0)
    location_score: float = Field(ge=0, le=10)


def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found")

    return joblib.load(MODEL_PATH)


@app.get("/")
def home():

    return {"message": "House Price Prediction API is running"}


@app.get("/health")
def health():

    return {"status": "healthy"}


@app.post("/predict")
def predict_price(data: HouseData):

    try:

        model_data = load_model()

        model = model_data["model"]

        preprocessor = model_data["preprocessor"]

        input_data = [[
            data.area,
            data.bedrooms,
            data.bathrooms,
            data.house_age,
            data.location_score
        ]]

        transformed_data = preprocessor.transform(input_data)

        prediction = model.predict(transformed_data)

        return {
            "predicted_price": round(float(prediction[0]), 2)
        }

    except FileNotFoundError as error:

        raise HTTPException(status_code=503, detail=str(error))

    except Exception:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )