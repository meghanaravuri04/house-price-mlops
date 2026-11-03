import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def load_data(file_path):
   try:
       logger.info("Loading dataset")
       df = pd.read_csv(file_path)
       logger.info("Dataset loaded successfully")
       return df
   except Exception:
       logger.exception("Error loading dataset")
       raise


def preprocess_data(file_path):
   try:
       df = load_data(file_path)
       logger.info("Checking missing values")

       X = df.drop("price", axis=1)
       y = df["price"]

       X_train, X_test, y_train, y_test = train_test_split(
           X,
           y,
           test_size=0.2,
           random_state=42,
       )

       preprocessing_pipeline = Pipeline([
           ("imputer", SimpleImputer(strategy="median")),
           ("scaler", StandardScaler()),
       ])

       X_train = preprocessing_pipeline.fit_transform(X_train)
       X_test = preprocessing_pipeline.transform(X_test)
       logger.info("Preprocessing completed")

       return X_train, X_test, y_train, y_test, preprocessing_pipeline
   except Exception:
       logger.exception("Preprocessing failed")
       raise
