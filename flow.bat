@echo off
SETLOCAL EnableDelayedExpansion
echo ===================================================
echo Creating House Price MLOps Project Structure...
echo ===================================================

:: Root Files
echo. > requirements.txt
echo. > params.yaml
echo. > dvc.yaml
echo. > dvc.lock
echo. > README.md
echo. > .env
echo. > .gitignore
echo. > .dockerignore
echo. > Dockerfile
echo. > docker-compose.yml
echo. > streamlit_app.py

:: GitHub Workflows
mkdir .github\workflows
echo. > .github\workflows\ci.yml

:: Data Directory
mkdir data
echo. > data\houses.csv

:: Src Directory
mkdir src
echo. > src\__init__.py
echo. > src\preprocess.py
echo. > src\train.py
echo. > src\evaluate.py
echo. > src\mlflow_connection.py
echo. > src\mlflow_experiments.py
echo. > src\register_model.py

:: Models Directory
mkdir models
echo. > models\model.pkl

:: Metrics Directory
mkdir metrics
echo. > metrics\metrics.json

:: App Directory
mkdir app
echo. > app\main.py

:: Tests Directory
mkdir tests
echo. > tests\test_api.py

echo ===================================================
echo Project structure created successfully!
echo ===================================================
pause
