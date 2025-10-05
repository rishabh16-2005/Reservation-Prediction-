from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import  DataProcessor
from src.components.model_training import ModelTraining
from config.path_config import *
from utils.common_functions import read_yaml

if __name__=='__main__':
    #1 Data Ingestion 
    con = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config=con)
    data_ingestion.run_all_process_in_one()

    #2 Data Processing
    processor = DataProcessor(raw_path=RAW_FILE_PATH,
                              processed_dir=PROCESSED_DIR,
                              config_path=CONFIG_PATH)
    processor.process()

    #3 Model Training
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    trainer.run()