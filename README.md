# House_Price_Predictor

A web based application that predicts house prices based on features like location, room count, and median income using a Random Forest Regressor.

## Features

- Predicts house prices based on essential house details like location, room count, and median income.
- Interactive user interface built with Streamlit for easy input and results visualization.
- Pre-trained Random Forest model with scaled feature inputs for accurate predictions.

## Files

- **`frontend.py`**: Contains the Streamlit app code for the user interface.
- **`house_price_prediction.py`**: Includes the data preprocessing, model training, saving, loading, and prediction logic.
- **`housing.csv`**: Dataset used to train the model. Includes details such as latitude, longitude, total rooms, total bedrooms, and median income.

## Requirements

- Python 3.7 or higher
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `streamlit`

Install dependencies using:
```bash
pip install -r requirements.txt
