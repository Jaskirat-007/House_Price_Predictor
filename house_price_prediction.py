import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Load and preprocess the predefined dataset
def load_and_preprocess_data(file_path="housing.csv"):
    """
    Load and preprocess the housing dataset.
    Only relevant columns are selected for training.
    """
    data = pd.read_csv(file_path)

    # Handle missing values for 'total_bedrooms'
    data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())

    # Define the target column
    target_column = "median_house_value"

    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)

    # Select essential features
    essential_features = ["latitude", "longitude", "total_rooms", "total_bedrooms", "median_income", target_column]
    data = data[essential_features]

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns.tolist()

# Train and save the Random Forest model
def train_and_save_model(model_save_path="model.pkl"):
    """
    Train a Random Forest model using simplified inputs.
    """
    # Preprocess the dataset
    X, y, scaler, feature_names = load_and_preprocess_data("housing.csv")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred),
    }

    print(f"Model trained successfully! Metrics: {metrics}")

    # Save the model
    with open(model_save_path, "wb") as model_file:
        pickle.dump((model, scaler, feature_names), model_file)

    return model, metrics

# Load the trained model
def load_model(model_path="model.pkl"):
    """
    Load a trained model from a file.
    """
    with open(model_path, "rb") as model_file:
        model, scaler, feature_names = pickle.load(model_file)
    return model, scaler, feature_names

# Predict house prices
def predict_price(model, scaler, feature_names, input_data):
    """
    Predict house prices based on user-friendly input.
    """
    # Convert user input into a DataFrame to match the model's feature set
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Scale the input features using the pre-fitted scaler
    input_scaled = scaler.transform(input_df)

    # Predict the price using the trained model
    prediction = model.predict(input_scaled)
    return prediction[0]
