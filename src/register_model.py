import logging

from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def register_model():
   try:
       client = MlflowClient()
       model_name = "HousePriceModel"
       versions = client.search_model_versions(f"name='{model_name}'")

       if not versions:
           logger.warning("No registered model found")
           return

       for version in versions:
           logger.info(
               "Model: %s | Version: %s | Status: %s",
               version.name,
               version.version,
               version.status,
           )
   except Exception:
       logger.exception("Model registry check failed")
       raise


if __name__ == "__main__":
   register_model()