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

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomExcep(e, sys)
    
def evaluate_model(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}

        for model_name, model_instance in models.items():
            # param_grid = params.get(model_name, {})
            # gs = GridSearchCV(model_instance, param_grid, cv=3)
            # gs.fit(X_train, y_train)

            # best_model = gs.best_estimator_
            model_instance.fit(X_train, y_train)

            y_train_pred = model_instance.predict(X_train)
            y_test_pred = model_instance.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomExcep(e, sys)
    
def load_obj(file_path):
    try:

        with open(file_path, 'rb') as file:
            return joblib.load(file)

    except Exception as e:
        raise CustomExcep(e, sys)