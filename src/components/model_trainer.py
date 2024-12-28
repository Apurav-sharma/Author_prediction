from src.exception import CustomExcep
from src.logger import logging

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import os, sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainingConfig:
    model_file_path = os.path.join("artifact", "model.pkl")


class ModelTraining:
    def __init__(self):
        self.modelConfig = ModelTrainingConfig()
    
    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Model Training has started")

            models = {
                # "Logistic Regression": LogisticRegression(),
                # "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100),
                # "SVM": SVC(kernel='linear'),
                # "KNN": KNeighborsClassifier(n_neighbors=5),
                # "Naive Bayes": GaussianNB(),
                "XGBClassifier": XGBClassifier(),
                "CatBoostClassifier": CatBoostClassifier(verbose=False),
                "AdaBoostClassifier": AdaBoostClassifier(n_estimators=100),
                # "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100)
            }

            hyperparameters = {
                # "Logistic Regression": {
                #     "C": [0.1, 1, 10],
                #     "solver": ["liblinear", "lbfgs"],
                #     "max_iter": [100, 200]
                # },
                # "Decision Tree": {
                #     "max_depth": [3, 5, 10],
                #     "min_samples_split": [2, 5, 10],
                #     "min_samples_leaf": [1, 2, 5],
                #     "criterion": ["gini", "entropy"]
                # },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 10],
                    # "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "criterion": ["gini", "entropy"]
                },
                # "SVM": {
                #     "C": [0.1, 1, 10],
                #     "kernel": ["rbf"]
                # },
                # "KNN": {
                #     "n_neighbors": [5, 7],
                #     "weights": ["uniform", "distance"],
                #     "algorithm": ["auto", "ball_tree", "brute"]
                # },
                # "Naive Bayes": {
                #     "var_smoothing": [1e-9, 1e-7]
                # },
                "XGBClassifier": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6]
                },
                "CatBoostClassifier": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [6, 8]
                },
                "AdaBoostClassifier": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                },
                # "GradientBoostingClassifier": {
                #     "n_estimators": [100, 200],
                #     "learning_rate": [0.01, 0.1],
                #     "max_depth": [3, 5]
                # }
            }

            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )

            logging.info("Data is splitted for training and testing")

            model_report:dict = evaluate_model(X_train, X_test, y_train, y_test, models = models, params = hyperparameters)

            logging.info("Training of model has executed")

            best_model = ""
            best_score = 0

            for key, value in model_report.items():
                if(value > best_score):
                    best_score = value
                    best_model = key

            selected_model = models[best_model]

            logging.info("Best Model has been selected")

            save_object(obj=selected_model, file_path=self.modelConfig.model_file_path)

            y_pred = selected_model.predict(X_test)
            logging.info("Model prediction for model has done")

            report = classification_report(y_test, y_pred)
            logging.info("Classification report for model has done")
            print(selected_model)

            return report

        except Exception as e:
            raise CustomExcep(e, sys)