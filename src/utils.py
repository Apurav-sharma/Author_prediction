import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomExcep
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib

# Function to save an object (e.g., model or preprocessor) to a file
def save_object(file_path, obj):
    """
    Save an object to a specified file path using dill.

    Args:
        file_path (str): The path where the object will be saved.
        obj: The object to be saved.

    Raises:
        CustomExcep: If an error occurs during saving.
    """
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the file in binary write mode
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomExcep(e, sys)

# Function to evaluate multiple models and return their test scores
def evaluate_model(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}

        # Loop through each model in the dictionary
        for model_name, model_instance in models.items():
            # Uncomment the following block to use GridSearchCV for hyperparameter tuning
            param_grid = params.get(model_name, {})
            gs = GridSearchCV(model_instance, param_grid, cv=3)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_

            # Fit the model with training data
            best_model.fit(X_train, y_train)

            # Make predictions for both training and testing sets
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate R2 scores for training and testing data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report dictionary
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomExcep(e, sys)

# Function to load a saved object (e.g., model or preprocessor) from a file
def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file:
            return joblib.load(file)
    except Exception as e:
        raise CustomExcep(e, sys)
