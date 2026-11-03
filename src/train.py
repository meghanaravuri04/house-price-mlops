import logging
import os

import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.preprocess import preprocess_data

logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def train_model():
   try:
       logger.info("Starting model training")

       with open("params.yaml", "r") as file:
           params = yaml.safe_load(file)

       data_path = params["data"]["path"]
       X_train, X_test, y_train, y_test, pipeline = preprocess_data(
           data_path
       )

       model = RandomForestRegressor(
           n_estimators=params["model"]["n_estimators"],
           max_depth=params["model"]["max_depth"],
           random_state=params["data"]["random_state"],
       )
       model.fit(X_train, y_train)

       os.makedirs("models", exist_ok=True)
       model_data = {
           "model": model,
           "preprocessor": pipeline,
       }
       joblib.dump(model_data, "models/model.pkl")

       logger.info("Model training completed")
       logger.info("Model saved successfully")
       return model, X_test, y_test
   except Exception:
       logger.exception("Model training failed")
       raise


if __name__ == "__main__":
   train_model()