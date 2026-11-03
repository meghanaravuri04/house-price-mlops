import logging
import os

import dagshub
import mlflow
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def connect_mlflow():
   try:
       dagshub.init(
           repo_owner=os.getenv("DAGSHUB_USERNAME"),
           repo_name=os.getenv("DAGSHUB_REPO_NAME"),
           mlflow=True,
       )

       mlflow.set_experiment("House-Price-Prediction")
       logger.info("DagsHub and MLflow connected")
   except Exception:
       logger.exception("MLflow connection failed")
       raise