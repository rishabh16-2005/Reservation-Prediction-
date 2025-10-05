import os 
import sys
import pandas as pd 
import pymongo
from config.path_config import *
from utils.common_functions import read_yaml
from sklearn.model_selection import train_test_split
from src.logger import MLLogger
from src.exception import CustomException
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')
logger = MLLogger(component_name='Data_Ingestion',log_dir='logs')

class DataIngestion:
    def __init__(self,config):
        self.config = config['data_ingestion']
        self.database_name = self.config['DATA_INGESTION_DATABASE_NAME']
        self.collection_name = self.config['DATA_INGESTION_COLLECTION_NAME']
        # self.train_split_ratio = self.config['DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO']

        logger.info('Created a Raw Directory for storing data in artifacts folder')
        os.makedirs(RAW_DIR,exist_ok=True)

    def download_csv_from_mongodb(self):
        try:
            logger.info('Connecting to MongoDB')
            self.client = pymongo.MongoClient(MONGO_DB_URL)
            logger.info('Successfully connected to Client')
            collection = self.client[self.database_name][self.collection_name]
            logger.info('Successfully got the collection')
            document = list(collection.find({}))
            logger.info('Successfully Fetched the document from collection')
            df = pd.DataFrame(document)
            if '_id' in df.columns:
                df.drop(columns=['_id'], inplace=True)
            df.to_csv(RAW_FILE_PATH,index=False)
            logger.info('Succesfully downloaded the raw csv data from mongo db')
            return RAW_FILE_PATH
        except Exception as e:
            raise CustomException(str(e),*sys.exc_info())
        
    # def split_data(self):
    #     try:
    #         logger.info('Starting the Splitting Process')
    #         data = pd.read_csv(RAW_FILE_PATH)
    #         train_data , test_data = train_test_split(data,test_size=1-self.train_split_ratio,random_state=42)

    #         train_data.to_csv(TRAIN_FILE_PATH,index=False)
    #         test_data.to_csv(TEST_FILE_PATH,index=False)

    #         logger.info(f'Train data saved to {TRAIN_FILE_PATH}')
    #         logger.info(f'Test data saved to  {TEST_FILE_PATH}')
    #         logger.info('SuccessFully Splitted the Data into train and testing') 
    #     except Exception as e:
    #         raise CustomException(str(e),*sys.exc_info())

    def run_all_process_in_one(self):
        try:
            logger.info('Starting data ingestion process')
            file_path = self.download_csv_from_mongodb()    
            # self.split_data()
            logger.info('Data Ingestion completed succesfully')
        except CustomException as e:
            logger.error(f'CustomException : {str(e)}')


if __name__=='__main__':
    con = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config=con)
    data_ingestion.run_all_process_in_one()