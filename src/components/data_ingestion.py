import sys
import os
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomExcep
from dataclasses import dataclass
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining


@dataclass
class dataIngestionConfig:
    train_data_path:str = os.path.join("artifact", "train.txt")
    test_data_path:str = os.path.join("artifact", "test.txt")
    raw_data_path:str = os.path.join("artifact", "data.txt")


class DataIngestion:
    def __init__(self):
        self.ingestionConfig = dataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has started")

        try:
            logging.info("Data has been taken as dataFrame")

            df_train = pd.read_csv("notebook\\data\\avila-tr.txt")
            df_test = pd.read_csv("notebook\\data\\avila-ts.txt")

            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path), exist_ok=True)

            df_train.to_csv(self.ingestionConfig.train_data_path, index=False, header = True)
            df_test.to_csv(self.ingestionConfig.test_data_path, index = False, header = True)

            logging.info("Train test splited")

            logging.info("data Ingestion step is completed")

            return (
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path
            )

        except Exception as e:
            raise CustomExcep(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()

    train_arr, test_arr, file_path = data_transformation.transform_data(train_path, test_path)

    model_trainer = ModelTraining()

    print(model_trainer.initiate_model_training(train_arr, test_arr))
