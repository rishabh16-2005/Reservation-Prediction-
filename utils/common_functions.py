import os 
import pandas as pd
from src.logger import MLLogger
from src.exception import CustomException
import yaml 
import sys
logger = MLLogger(component_name='utils-common-functions',log_dir='logs')

def read_yaml(file_path : str):
    """
    Reads a YAML file from the given path and returns the content as a dictionary.
    Raises:
        CustomException : If the file doesn't exist or reading fails.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File is Not in the given path')
        
        with open(file_path,'r') as y_file:
            config = yaml.safe_load(y_file)
            logger.info('Succesfully read the YAML file')
            return config
        
    except Exception as e:
        logger.error(f'Error while reading YAML file {file_path}')
        raise CustomException(str(e),*sys.exc_info())
    
def load_data(file_path:str):
    """
    Reads a csv file from the given file path and 
    return the data as a pandas DataFrame
    """
    try:
        logger.info('loading data')
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f'Error loading the data {e}')
        raise CustomException(str(e),*sys.exc_info())

