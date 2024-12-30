from src.exception import CustomExcep
from src.logger import logging

from sklearn.metrics import classification_report
import os, sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainingConfig:
    """
    Configuration class for model training.
    Stores the file path where the trained model will be saved.
    """
    model_file_path = os.path.join("artifact", "model.pkl")


class ModelTraining:
    """
    Class to handle the training of machine learning models.
    """
    def __init__(self):
        self.modelConfig = ModelTrainingConfig()
    
    def initiate_model_training(self, train_arr, test_arr):
        """
        Initiates the model training process.

        Args:
            train_arr (np.ndarray): Training data array with features and target.
            test_arr (np.ndarray): Testing data array with features and target.

        Returns:
            str: Classification report for the best selected model.

        Raises:
            CustomExcep: For any exception that occurs during the process.
        """
        try:
            logging.info("Model Training has started")

            # Define the models to train
            models = {
                "CatBoostClassifier": CatBoostClassifier(verbose=False),
            }

            # Define hyperparameters for GridSearchCV (if needed)
            hyperparameters = {
                "CatBoostClassifier": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [6, 8]
                }
            }

            # Split the data into features and target
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],  # Training features
                test_arr[:, :-1],   # Testing features
                train_arr[:, -1],   # Training target
                test_arr[:, -1]     # Testing target
            )

            logging.info("Data is split for training and testing")

            # Evaluate models and get their performance
            model_report: dict = evaluate_model(X_train, X_test, y_train, y_test, models=models, params=hyperparameters)

            logging.info("Training of models has been executed")

            # Select the best model based on performance
            best_model = ""
            best_score = 0
            for key, value in model_report.items():
                if value > best_score:
                    best_score = value
                    best_model = key

            selected_model = models[best_model]

            logging.info("Best Model has been selected")

            # Fit the best model on the training data
            selected_model.fit(X_train, y_train)

            # Save the trained model to a file
            save_object(obj=selected_model, file_path=self.modelConfig.model_file_path)

            # Make predictions on the test data
            y_pred = selected_model.predict(X_test)
            logging.info("Model prediction for test data is done")

            # Generate and return a classification report
            report = classification_report(y_test, y_pred)
            logging.info("Classification report for the model is completed")
            print(report)

            return report

        except Exception as e:
            raise CustomExcep(e, sys)
