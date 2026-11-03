import logging
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")

st.write("Enter house details to predict the price.")

area = st.number_input(
    "Area (sq.ft)",
    min_value=100,
    value=1200
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=1
)

house_age = st.number_input(
    "House Age",
    min_value=0,
    max_value=100,
    value=5
)

location_score = st.slider(
    "Location Score",
    min_value=0,
    max_value=10,
    value=7
)

if st.button("Predict House Price"):

    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_age": house_age,
        "location_score": location_score
    }

    try:

        response = requests.post(
            API_URL,
            json=input_data,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        st.success("Prediction completed!")

        st.metric(
            "Predicted House Price",
            f"₹ {result['predicted_price']:,.2f}"
        )

    except requests.RequestException:
        logger.exception("API connection failed")

        st.error(
            "Unable to connect to FastAPI. "
            "Please check whether the API is running."
        )
