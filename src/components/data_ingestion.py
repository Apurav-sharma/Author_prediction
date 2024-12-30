import sys
import os
import pandas as pd
import logging
from src.exception import CustomExcep
from src.config import dataIngestionConfig
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining


class DataIngestion:
    def __init__(self):
        # Initialize the data ingestion configuration
        self.ingestionConfig = dataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has started")

        try:
            logging.info("Data has been taken as dataFrame")

            # Read the training and testing data from CSV files
            df_train = pd.read_csv("notebook\\data\\avila-tr.txt")
            df_test = pd.read_csv("notebook\\data\\avila-ts.txt")

            # Create directories for storing the ingested data if they don't exist
            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path), exist_ok=True)

            # Save the training and testing data to the specified paths
            df_train.to_csv(self.ingestionConfig.train_data_path, index=False, header=True)
            df_test.to_csv(self.ingestionConfig.test_data_path, index=False, header=True)

            logging.info("Train test split completed")
            logging.info("Data Ingestion step is completed")

            # Return the paths to the ingested training and testing data
            return (
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path
            )

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomExcep(e, sys)

if __name__ == "__main__":
    # Create an instance of DataIngestion and initiate the data ingestion process
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()

    train_arr, test_arr, file_path = data_transformation.transform_data(train_path, test_path)

    model_trainer = ModelTraining()

    print(model_trainer.initiate_model_training(train_arr, test_arr))
