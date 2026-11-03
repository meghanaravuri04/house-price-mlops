import logging

import mlflow
import mlflow.sklearn

from src.evaluate import evaluate_model
from src.mlflow_connection import connect_mlflow
from src.train import train_model

logger = logging.getLogger(__name__)


def run_experiment():
   try:
       connect_mlflow()

       with mlflow.start_run() as run:
           model, _, _ = train_model()
           metrics = evaluate_model()

           mlflow.log_param("model", "RandomForestRegressor")
           mlflow.log_param("n_estimators", 100)
           mlflow.log_param("max_depth", 10)
           mlflow.log_metrics(metrics)

           mlflow.sklearn.log_model(
               sk_model=model,
               artifact_path="house_price_model",
               registered_model_name="HousePriceModel",
           )

           logger.info("MLflow run completed: %s", run.info.run_id)
   except Exception:
       logger.exception("MLflow experiment failed")
       raise


if __name__ == "__main__":
   run_experiment()