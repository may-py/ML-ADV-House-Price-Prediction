import os
import sys
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split




@dataclass(frozen=True)
class DataIngesionConfig:
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    train_data_path:str = os.path.join('artifacts','train.csv')


class DataIngesion:
    def __init__(self):
        self.ingesoin_config = DataIngesionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            df = pd.read_csv("notebook\\data\\train.csv")
            os.makedirs(os.path.dirname(self.ingesoin_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingesoin_config.raw_data_path,index=False, header=True)

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingesoin_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingesoin_config.test_data_path,index=False, header=True)

            return(
                self.ingesoin_config.train_data_path,
                self.ingesoin_config.test_data_path
            )

        except Exception as e:
            raise e
        

if __name__ == '__main__':
    data_ingestion = DataIngesion()
    data_ingestion.initiate_data_ingestion()