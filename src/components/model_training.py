import os 
import sys 
import joblib
import pandas as pd 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from src.logger import MLLogger
from src.exception import CustomException
from config.path_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint
from lightgbm import LGBMClassifier
import mlflow

logger = MLLogger(component_name='Model Training',log_dir='logs')

class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        
    def load_and_split_data(self):
        try:
            logger.info(f'Loading data From {self.train_path}')
            train_df = load_data(self.train_path)

            logger.info(f'Loading data From {self.test_path}')
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logger.info('Splitted the data for Model Training')
            return X_train,y_train,X_test,y_test
        except Exception as e:
            logger.error('Error Occured while loading and splitting data')
            raise CustomException(str(e),*sys.exc_info())
        
    def train_lgbm(self,X_train,y_train):
        try:
            logger.info('Model Initialisation')

            lgbm_model = LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info('Starting our HyperParameter Tuning')
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter= self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )
            random_search.fit(X_train,y_train)

            logger.info('Completed Hyperparameter Tuning')

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f'Best Parameters are : {best_params}')

            return best_lgbm_model
        except Exception as e:
            logger.info('Error while Hyperparameter Tuning')
            raise CustomException(str(e),*sys.exc_info())
        
        
    def evaluate_model(self,model:LGBMClassifier,X_test,y_test):
        try:
            logger.info('Evaluating the Model')
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)

            logger.info(f'Accuracy Score : {accuracy}')
            logger.info(f'Precision Score : {precision}')
            logger.info(f'f1 Score : {f1}')
            logger.info(f'recall Score : {recall}')
            
            return {
                'accuracy':accuracy,
                'precision':precision,
                'f1':f1,
                'recall':recall
            }
        except Exception as e:
            logger.error('Failed to evaluate model')
            raise CustomException (str(e),*sys.exc_info())
        
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            logger.info('Saving the Model')
            joblib.dump(model,self.model_output_path)
            logger.info(f'Model Saved to {self.model_output_path}')
        except Exception as e:
            logger.error('Error while saving the Model')
            raise CustomException(str(e),*sys.exc_info())
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info('Starting our Model Training Pipeline')
                logger.info('Logging the training and testing dataset to MLFLOW')
                mlflow.log_artifact(self.train_path,artifact_path='datasets')
                mlflow.log_artifact(self.test_path,artifact_path='datasets')

                X_train , y_train , X_test,y_test = self.load_and_split_data()
                best_model = self.train_lgbm(X_train,y_train)
                evaluation_metrics = self.evaluate_model(best_model,X_test,y_test)
                self.save_model(best_model)
                logger.info('Logging the Model into MLFLOW')
                mlflow.log_artifact(self.model_output_path,artifact_path='model')

                logger.info('Logging Params and Metrics to MLFLOW')
                mlflow.log_params(best_model.get_params())
                mlflow.log_metrics(evaluation_metrics)

                logger.info('Model Training successfully Completed')
        except Exception as e:
            logger.error('Error while running the pipeline')
            raise CustomException(str(e),*sys.exc_info())
        
if __name__=='__main__':
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    trainer.run()