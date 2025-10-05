import pandas as pd 
import os
import sys 
import numpy as np 
from src.logger import MLLogger
from src.exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2 , f_classif
from sklearn.model_selection import train_test_split
logger = MLLogger(component_name='Data_PreProcessing',log_dir='logs')

class DataProcessor:
    def __init__(self, raw_path, processed_dir, config_path):
        self.raw_path = raw_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        self.train_split_ratio = self.config['data_processing']['DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO']
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self, df: pd.DataFrame):
        try:
            logger.info('Starting data preprocessing')

            # Drop unnecessary columns
            df.drop(columns=['Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)

            categorical_columns = self.config['data_processing']['categorical_columns']
            numerical_columns = self.config['data_processing']['numerical_columns']

            # Label Encoding
            logger.info('Applying LabelEncoding for categorical features')
            label_encoder = LabelEncoder()
            for col in categorical_columns:
                df[col] = label_encoder.fit_transform(df[col])

            # Handle skewness
            skew_threshold = self.config['data_processing']['skewness_threshold']
            col = ['lead_time','avg_price_per_room']
            skewness = df[col].apply(lambda x: x.skew())
            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df
        except Exception as e:
            logger.error('Error in data preprocessing')
            raise CustomException(str(e), sys)

    def split_data(self, df: pd.DataFrame):
        try:
            logger.info('Splitting dataset into train and test')
            X = df.drop(columns=self.config['data_processing']['target_column'])
            y = df[self.config['data_processing']['target_column']]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1 - self.train_split_ratio,
                random_state=42, stratify=y
            )

            logger.info('Split completed')
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(str(e), sys)

    def balance_data(self, X_train, y_train):
        try:
            logger.info('Handling imbalanced data using SMOTE')
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            logger.info('Data balanced successfully')
            return X_res, y_res
        except Exception as e:
            logger.error('Error while balancing data')
            raise CustomException(str(e), sys)

    def select_features(self, X, y):
        try:
            logger.info('Starting feature selection')

            # Categorical features (Chi2)
            cat_cols = [col for col in self.config['data_processing']['categorical_columns']
                        if col != self.config['data_processing']['target_column']]
            X_cat = X[cat_cols]
            chi_selector = SelectKBest(score_func=chi2, k='all')
            chi_selector.fit(X_cat, y)
            chi_scores = pd.DataFrame({
                'Feature': X_cat.columns,
                'Chi2 Score': chi_selector.scores_
            }).sort_values(by='Chi2 Score', ascending=False)
            top_chi = self.config['data_processing']['top_features_for_Chi2']
            top_categorical_features = chi_scores['Feature'].head(top_chi).tolist()

            # Numerical features (ANOVA)
            num_cols = self.config['data_processing']['numerical_columns']
            X_num = X[num_cols]
            anova_selector = SelectKBest(score_func=f_classif, k='all')
            anova_selector.fit(X_num, y)
            anova_scores = pd.DataFrame({
                'Feature': X_num.columns,
                'F Score': anova_selector.scores_
            }).sort_values(by='F Score', ascending=False)
            top_anova = self.config['data_processing']['top_features_for_Anova']
            top_numerical_features = anova_scores['Feature'].head(top_anova).tolist()

            selected_features = top_categorical_features + top_numerical_features
            X_selected = pd.concat([X_cat[top_categorical_features], X_num[top_numerical_features]], axis=1) 
            X_selected['booking_status'] = y 
            logger.info('Concatenated the Selected features') 
            return X_selected.drop(columns=['arrival_year','arrival_month','arrival_date'])
        except Exception as e:
            logger.error('Error during feature selection')
            raise CustomException(str(e), sys)

    def save_data(self, df: pd.DataFrame, file_path: str):
        try:
            logger.info(f'Saving processed data to {file_path}')
            df.to_csv(file_path, index=False)
            logger.info('Data saved successfully')
        except Exception as e:
            logger.error('Error while saving data')
            raise CustomException(str(e), sys)

    def process(self):
        try:
            logger.info('Loading raw data')
            df = load_data(self.raw_path)
            df = self.preprocess_data(df)

            X_train, X_test, y_train, y_test = self.split_data(df)
            # X_train_bal, y_train_bal = self.balance_data(X_train, y_train)
            X_train_sel = self.select_features(X_train, y_train)
            feature_cols = X_train_sel.drop(columns=[self.config['data_processing']['target_column']]).columns
            X_test_sel = X_test[feature_cols]

            # Combine features and target for saving
            train_df = X_train_sel.copy()
            train_df[self.config['data_processing']['target_column']] = y_train
            test_df = X_test_sel.copy()
            test_df[self.config['data_processing']['target_column']] = y_test

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info('Data processing completed successfully')
        except Exception as e:
            logger.error('Error during full data processing')
            raise CustomException(str(e), sys)


if __name__ == '__main__':
    processor = DataProcessor(raw_path=RAW_FILE_PATH,
                              processed_dir=PROCESSED_DIR,
                              config_path=CONFIG_PATH)
    processor.process()
