import os
import sys
from src.exception import CustomExcep
from src.utils import load_obj
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:

            model_path = r'C:\Users\Apurav\OneDrive\Desktop\Machine learning\artifact\model.pkl'
            preprocessor_path = r'C:\Users\Apurav\OneDrive\Desktop\Machine learning\artifact\preprocessor.pkl'
            

            model = load_obj(model_path)
            preprocessor = load_obj(preprocessor_path)

            data_scaled = preprocessor.transform(features)

            res = model.predict(data_scaled)
            return res

        except Exception as e:
            raise CustomExcep(e, sys)

class CustomData:
    def __init__(self, intercolumnar_distance: float, upper_margin: float, lower_margin: float, exploitation: float, row_number: float, 
        modular_ratio: float, interlinear_spacing: float, weight: float, peak_number: float, modular_ratio_by_interlinear_spacing: float
        ):

        self.intercolumnar_distance = intercolumnar_distance
        self.upper_margin = upper_margin
        self.lower_margin = lower_margin
        self.exploitation = exploitation
        self.row_number = row_number
        self.modular_ratio = modular_ratio
        self.interlinear_spacing = interlinear_spacing
        self.weight = weight
        self.peak_number = peak_number
        self.modular_ratio_by_interlinear_spacing = modular_ratio_by_interlinear_spacing

    def get_data_frame(self):
        try:
            
            custom_data = {
                "intercolumnar_distance": [self.intercolumnar_distance],
                "upper_margin": [self.upper_margin],
                "lower_margin": [self.lower_margin],
                "exploitation": [self.exploitation],
                "row_number": [self.row_number],
                "modular_ratio": [self.modular_ratio],
                "interlinear_spacing": [self.interlinear_spacing],
                "weight": [self.weight],
                "peak_number": [self.peak_number],
                "modular_ratio_by_interlinear_spacing": [self.modular_ratio_by_interlinear_spacing]
            }

            return pd.DataFrame(custom_data)

        except Exception as e:
            raise CustomExcep(e, sys)