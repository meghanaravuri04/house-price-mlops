import json
import logging
import os

import joblib
import yaml
from sklearn.metrics import (
   mean_absolute_error,
   mean_squared_error,
   r2_score,
)

from src.preprocess import preprocess_data

logger = logging.getLogger(__name__)


def evaluate_model():
   try:
       logger.info("Starting model evaluation")

       with open("params.yaml", "r") as file:
           params = yaml.safe_load(file)

       _, X_test, _, y_test, _ = preprocess_data(
           params["data"]["path"]
       )

       model_data = joblib.load("models/model.pkl")
       model = model_data["model"]
       predictions = model.predict(X_test)

       mse = mean_squared_error(y_test, predictions)
       metrics = {
           "MAE": mean_absolute_error(y_test, predictions),
           "MSE": mse,
           "RMSE": mse ** 0.5,
           "R2_Score": r2_score(y_test, predictions),
       }

       os.makedirs("metrics", exist_ok=True)
       with open("metrics/metrics.json", "w") as file:
           json.dump(metrics, file, indent=4)

       logger.info("Evaluation completed: %s", metrics)
       return metrics

   except Exception:
       logger.exception("Model evaluation failed")
       raise


if __name__ == "__main__":
   evaluate_model()