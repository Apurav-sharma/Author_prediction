from src.logger import logging
from src.exception import CustomExcep

import sys
import os
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTranformationConfig:
    file_path: str = os.path.join("artifact", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTranformationConfig()
    
    def get_data_tranformation_obj(self):
        try:
            pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("standard", StandardScaler())
                ]
            )
            logging.info("Pipeline object created successfully.")
            return pipeline
        
        except Exception as e:
            raise CustomExcep(e, sys)
    
    def transform_data(self, train_path, test_path):
        try:
            column_names = [
            'intercolumnar_distance', 'upper_margin', 'lower_margin', 'exploitation', 'row_number', 'modular_ratio', 'interlinear_spacing', 'weight', 'peak_number', 'modular_ratio_by_interlinear_spacing', 'Class'
            ]

            df_train = pd.read_csv(train_path, header=None, names = column_names)
            df_test = pd.read_csv(test_path, header=None, names = column_names)

            logging.info("Train and test data loaded successfully.")
            
            target_column = "Class"
            X_train = df_train.drop(columns=[target_column], axis=1)
            y_train = df_train[target_column].map({
                'A': 0,
                'B': 1,
                'C': 2,
                'D': 3,
                'E': 4,
                'F': 5,
                'G': 6,
                'H': 7,
                'I': 8,
                'W': 9,
                'X': 10,
                'Y': 11,
            })

            X_test = df_test.drop(columns=[target_column], axis=1)
            y_test = df_test[target_column].map({
                'A': 0,
                'B': 1,
                'C': 2,
                'D': 3,
                'E': 4,
                'F': 5,
                'G': 6,
                'H': 7,
                'I': 8,
                'W': 9,
                'X': 10,
                'Y': 11,
            })

            logging.info("Separated features and target variable.")
            
            pipeline = self.get_data_tranformation_obj()

            X_train_transformed = pipeline.fit_transform(X_train)
            X_test_transformed = pipeline.transform(X_test)

            logging.info("Data transformation completed.")
            
            train_arr = np.c_[X_train_transformed, np.array(y_train)]

            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            save_object(
                file_path=self.data_transform_config.file_path,
                obj=pipeline
            )

            logging.info("preprocessor object saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transform_config.file_path
            )

        except Exception as e:
            raise CustomExcep(e, sys)
