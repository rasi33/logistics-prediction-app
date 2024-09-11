import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained Random Forest model and feature list
model = joblib.load('random_forest_model.pkl')
feature_list = joblib.load('feature_list.pkl')

# Streamlit App title
st.title("Logistics Delivery Time Prediction Tool")

# Introduction Section
st.write("""
## Welcome to the Logistics Delivery Time Prediction Tool!

This tool helps you predict the delivery time for shipments based on various factors such as the origin port, carrier, service level, and weight. 

### Instructions:
1. **Select Origin Port:** Choose the port where the shipment originates.
2. **Select Carrier:** Choose the carrier responsible for the shipment.
3. **Select Service Level:** Choose the type of delivery service.
   - **DTD**: Door-to-Door
   - **DTP**: Door-to-Port
   - **CRF**: Cost and Freight
4. **Enter Weight (in KG):** Provide the weight of the shipment in kilograms.

After filling in all the details, click on the 'Predict Delivery Time' button to get the estimated delivery time in days.
""")

# Input features for prediction
origin_port = st.selectbox("Select Origin Port:", ['PORT04', 'PORT05', 'PORT06'], help="Choose the port where the shipment originates.")  # Example options
carrier = st.selectbox("Select Carrier:", ['V444_0', 'V444_1', 'V444_2'], help="Select the carrier responsible for the shipment.")
service_level = st.selectbox("Select Service Level:", ['DTD', 'DTP', 'CRF'], help="Select the type of delivery service (e.g., DTD: Door-to-Door).")
weight = st.number_input("Enter Weight (in KG):", min_value=0.0, help="Enter the weight of the shipment in kilograms (e.g., 11.20).")

# Prepare input data in the required format
input_data = {
    'Origin Port_PORT04': 1 if origin_port == 'PORT04' else 0,
    'Origin Port_PORT05': 1 if origin_port == 'PORT05' else 0,
    'Carrier_V444_0': 1 if carrier == 'V444_0' else 0,
    'Carrier_V444_1': 1 if carrier == 'V444_1' else 0,
    'Service Level_DTD': 1 if service_level == 'DTD' else 0,
    'Weight': weight,
    # Add other features similarly...
}

# Ensure all features used during training are present in the input data
input_df = pd.DataFrame([input_data], columns=feature_list).fillna(0)  # Fill missing features with 0

# Predict delivery time
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Delivery Time: {prediction[0]:.2f} days")
