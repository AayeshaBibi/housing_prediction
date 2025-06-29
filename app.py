import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration for Streamlit App ---
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- Load your trained model ---
# Ensure 'newmodel.pkl' is in the same directory as this 'app.py' file.
try:
    # 'newmodel' should contain your entire pipeline (preprocessor + model)
    model = joblib.load("newmodel.pkl")
    st.success("Machine learning model loaded successfully! Ready for predictions.")
except FileNotFoundError:
    st.error("Error: Model file 'newmodel.pkl' not found.")
    st.info("Please ensure you have run 'ModelComparisonTable.ipynb' and it successfully saved 'newmodel.pkl' in the same directory as this Streamlit app.")
    st.stop() # Stop execution if model isn't found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop() # Stop execution for other loading errors

st.title("California Housing Price Predictor 🏠")
st.write("Enter the characteristics of a house in California to get a predicted median house value.")

# --- Input Fields for Features ---
st.header("House Characteristics")

# Numerical Inputs
# Using broader ranges to accommodate various inputs
longitude = st.slider("Longitude", -125.0, -114.0, -118.0, 0.001, help="Geographic longitude of the house.")
latitude = st.slider("Latitude", 32.0, 42.0, 34.0, 0.001, help="Geographic latitude of the house.")
housing_median_age = st.slider("Housing Median Age", 1, 55, 30, 1, help="Median age of the house within a block.")
total_rooms = st.number_input("Total Rooms", min_value=1, max_value=40000, value=2500, step=1, help="Total number of rooms in the block.")
total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=7000, value=500, step=1, help="Total number of bedrooms in the block (can be missing).")
population = st.number_input("Population", min_value=3, max_value=40000, value=1500, step=1, help="Total population in the block.")
households = st.number_input("Households", min_value=1, max_value=7000, value=450, step=1, help="Total number of households in the block.")
median_income = st.slider("Median Income (in tens of thousands $)", 0.5, 15.0, 5.0, step=0.1, format="%.2f", help="Median income for households in the block (e.g., 5.0 means $50,000).")

# Categorical Input
ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ocean_proximity_options,
    index=ocean_proximity_options.index('INLAND'), # Default to 'INLAND'
    help="Proximity to the ocean."
)

st.write("---")

# --- Make Prediction ---
if st.button("Predict House Value", help="Click to get the predicted median house value."):
    try:
        # Create a DataFrame for prediction.
        # It's crucial that the column names and their order exactly match
        # the features used during model training and preprocessing.
        input_data = pd.DataFrame([[
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income,
            ocean_proximity
        ]], columns=[
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity'
        ])

        # The loaded 'model' (which is your pipeline) will handle preprocessing
        # and then make the prediction.
        prediction = model.predict(input_data)[0]
        st.success(f"## Predicted Median House Value: ${prediction:,.2f}")
        st.balloons() # Just for fun!
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all inputs are valid and that the loaded model can process them. Check the console for more details.")

st.sidebar.markdown("### About This App")
st.sidebar.markdown(
    "This application demonstrates a machine learning model's ability to predict "
    "California housing prices based on various attributes. The model was trained "
    "using data from the California Housing dataset (or a synthetic dataset if not provided)."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit and Scikit-learn.")