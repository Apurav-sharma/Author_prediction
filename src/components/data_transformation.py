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

# Configuration class for storing the file path of the preprocessor
@dataclass
class DataTranformationConfig:
    file_path: str = os.path.join("artifact", "preprocessor.pkl")

# Class responsible for data transformation
class DataTransformation:
    def __init__(self):
        # Initialize configuration for data transformation
        self.data_transform_config = DataTranformationConfig()
    
    # Method to create and return a preprocessing pipeline object
    def get_data_tranformation_obj(self):
        try:
            # Define a pipeline with a median imputer and standard scaler
            pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),  # Handle missing values
                    ("standard", StandardScaler())  # Scale features to standard normal distribution
                ]
            )
            logging.info("Pipeline object created successfully.")
            return pipeline
        
        except Exception as e:
            # Raise custom exception in case of errors
            raise CustomExcep(e, sys)
    
    # Method to apply transformation on train and test data
    def transform_data(self, train_path, test_path):
        try:
            # Define column names for the dataset
            column_names = [
                'intercolumnar_distance', 'upper_margin', 'lower_margin', 'exploitation', 
                'row_number', 'modular_ratio', 'interlinear_spacing', 'weight', 
                'peak_number', 'modular_ratio_by_interlinear_spacing', 'Class'
            ]

            # Load training and testing data
            df_train = pd.read_csv(train_path, header=None, names=column_names)
            df_test = pd.read_csv(test_path, header=None, names=column_names)
            logging.info("Train and test data loaded successfully.")
            
            # Define the target column
            target_column = "Class"

            # Separate features and target variable for training data
            X_train = df_train.drop(columns=[target_column], axis=1)
            y_train = df_train[target_column].map({
                'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 
                'G': 6, 'H': 7, 'I': 8, 'W': 9, 'X': 10, 'Y': 11
            })

            # Separate features and target variable for testing data
            X_test = df_test.drop(columns=[target_column], axis=1)
            y_test = df_test[target_column].map({
                'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 
                'G': 6, 'H': 7, 'I': 8, 'W': 9, 'X': 10, 'Y': 11
            })

            logging.info("Separated features and target variable.")
            
            # Get the preprocessing pipeline
            pipeline = self.get_data_tranformation_obj()

            # Transform training and testing features using the pipeline
            X_train_transformed = pipeline.fit_transform(X_train)
            X_test_transformed = pipeline.transform(X_test)
            logging.info("Data transformation completed.")
            
            # Combine transformed features and target variable for training data
            train_arr = np.c_[X_train_transformed, np.array(y_train)]

            # Combine transformed features and target variable for testing data
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save the preprocessor object to a file
            save_object(
                file_path=self.data_transform_config.file_path,
                obj=pipeline
            )
            logging.info("Preprocessor object saved successfully")

            # Return the transformed arrays and the file path of the preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transform_config.file_path
            )

        except Exception as e:
            # Raise custom exception in case of errors
            raise CustomExcep(e, sys)
