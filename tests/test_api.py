from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home():
   response = client.get("/")

   assert response.status_code == 200
   assert "message" in response.json()


def test_health():
   response = client.get("/health")

   assert response.status_code == 200
   assert response.json()["status"] == "healthy"


def test_invalid_prediction():
   response = client.post(
       "/predict",
       json={
           "area": -100,
           "bedrooms": 2,
           "bathrooms": 1,
           "house_age": 5,
           "location_score": 7,
       },
   )

   assert response.status_code == 422
