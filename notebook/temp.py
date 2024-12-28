import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import streamlit as st

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and Preprocess Data
def load_data(file_path):
    """Load dataset from a CSV file."""
    logging.info("Loading dataset...")
    df = pd.read_csv(file_path)
    logging.info(f"Dataset loaded with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the dataset."""
    logging.info("Preprocessing dataset...")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    logging.info("Dataset preprocessing completed.")
    return X, y, scaler

# Train Model
def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    logging.info("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy}")
    logging.info(f"Classification Report:\n{report}")
    return accuracy, report

# Save Model and Scaler
def save_model(model, scaler, model_path, scaler_path):
    """Save model and scaler to disk."""
    logging.info("Saving model and scaler...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info("Model and scaler saved.")

# Streamlit App
def main():
    st.title("Scribe Classification System")
    st.write("Predict the author based on manuscript features")

    # Input fields for 10 features
    features = []
    for i in range(1, 11):
        value = st.number_input(f"Feature {i}", value=0.0)
        features.append(value)

    # Load scaler and model
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Model and scaler files not found. Train the model first.")
        return

    # Prediction
    if st.button("Predict Author"):
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)
        st.success(f"Predicted Author: {(prediction)}")

if __name__ == '__main__':
    # Main script
    dataset_path = 'data/avila-tr.txt'
    model_path = 'model.pkl'
    scaler_path = 'scaler.pkl'

    if st.sidebar.button("Train Model"):
        df = load_data(dataset_path)
        X, y, scaler = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

        save_model(model, scaler, model_path, scaler_path)

    main()
