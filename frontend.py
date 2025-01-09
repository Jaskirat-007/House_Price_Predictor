import streamlit as st
from house_price_prediction import load_model, predict_price

st.title("House Price Prediction App")

# Load the pre-trained model
model, scaler, feature_names = load_model()

# Sidebar for user inputs
st.sidebar.header("Enter House Details")
user_input = {}

# Only accept simplified inputs
for feature in feature_names:
    if feature in ["latitude", "longitude", "total_rooms", "total_bedrooms", "median_income"]:
        user_input[feature] = st.sidebar.number_input(f"{feature.capitalize()}", value=0.0)

# Predict the house price
if st.button("Predict Price"):
    try:
        price = predict_price(model, scaler, feature_names, user_input)
        st.success(f"The predicted house price is: ${price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
