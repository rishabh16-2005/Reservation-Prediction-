import os 
import sys 
import io
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,LeakyReLU
from keras.losses import BinaryCrossentropy
from keras.optimizers import AdamW
from keras import regularizers
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from keras.initializers import HeNormal,GlorotUniform
from src.logger import MLLogger
from src.exception import CustomException
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from config.path_config import PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,ANN_OUTPUT_PATH
from utils.common_functions import load_data
from sklearn.preprocessing import StandardScaler
import mlflow


logger = MLLogger(component_name='ANN_LOGS',log_dir='logs')

class ANNTraining:

    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        if not os.path.exists(self.model_output_path):
            os.makedirs(self.model_output_path,exist_ok=True)

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

            col = ['lead_time','avg_price_per_room']
            scaler = StandardScaler()
            X_train[col] = scaler.fit_transform(X_train[col])
            X_test[col] = scaler.transform(X_test[col])

            logger.info('Splitted the data for Model Training')
            return X_train,y_train,X_test,y_test
        except Exception as e:
            logger.error('Error Occured while loading and splitting data')
            raise CustomException(str(e),sys)

    def create_model_structure(self,input_size):
        try:
            l2_lambda = 0.0001
            logger.info('Model Initialisation')
            model = Sequential()

            # First Hidden Layer 
            model.add(Dense(
                units=512,
                kernel_initializer=HeNormal(),
                input_dim=input_size,
                kernel_regularizer=regularizers.l2(l2_lambda)
            ))
            model.add(BatchNormalization())
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(Dropout(0.1))

            ## Second Hidden Layer 
            model.add(Dense(
                units=256,
                kernel_initializer=HeNormal(),
                kernel_regularizer=regularizers.l2(l2_lambda)
            ))
            model.add(BatchNormalization())
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(Dropout(0.1))
            # Third Hidden Layer
            model.add(Dense(
                units=256,
                kernel_initializer=HeNormal(),
                kernel_regularizer=regularizers.l2(l2_lambda)
            ))
            model.add(BatchNormalization())
            model.add(LeakyReLU(negative_slope=0.01))
            model.add(Dropout(0.1))

            # Fourth Hidden Layer
            model.add(Dense(
                units=64,
                kernel_initializer=HeNormal(),
                kernel_regularizer=regularizers.l2(l2_lambda)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.05))
            model.add(LeakyReLU(negative_slope=0.01))
            # Output Layer
            model.add(Dense(
                units=1,
                activation='sigmoid',
                kernel_initializer=GlorotUniform()
            ))

            logger.info('Compiling the Model')
            model.compile(
                optimizer=AdamW(learning_rate=1e-3,weight_decay=1e-5,clipnorm=1.0),
                loss=BinaryCrossentropy(),
                metrics=['accuracy'],
                
            )
            
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + "\n"))
            summary_str = stream.getvalue()
        
            logger.info(f'Model Summary :\n {summary_str}')

            return model
        except Exception as e:
            logger.error('Error Occured while Creating Model Structure')
            raise CustomException(str(e),sys)
        
    def train_ann(self,X_train,y_train,model:Sequential):
        try:
            logger.info('Training Neural Network')
            early_stop = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
                )
            checkpoint = ModelCheckpoint('best_model.keras',
                                         monitor='val_loss',
                                         save_best_only=True)
            history = model.fit(X_train,y_train,epochs=500,verbose=1,batch_size=128,callbacks=[early_stop,lr_scheduler,checkpoint],validation_split=0.2)
            logger.info(f"Final Training Accuracy : {history.history['accuracy'][-1]:.4f}")
            logger.info(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
            logger.info('Model Training Completed')
        except Exception as e:
            logger.error('Error while Traning Neural Network')
            raise CustomException(str(e),sys)
        
    def evaluate_ann(self,X_test,y_test,model:Sequential):
        try:
            logger.info('Predicting on test Dataset')
            y_pred = (model.predict(X_test) > 0.3).astype("int32").flatten()
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            matrix = confusion_matrix(y_test,y_pred)
            logger.info(f'Accuracy Score : {accuracy}')
            logger.info(f'Precision Score : {precision}')
            logger.info(f'f1 Score : {f1}')
            logger.info(f'recall Score : {recall}')
            logger.info(f'Classification Matrix :\n {matrix}')
            return {
                    'accuracy':accuracy,
                    'precision':precision,
                    'f1':f1,
                    'recall':recall,
                    'confusion_matrix':matrix
                }

        except Exception as e:
            logger.error('Failed to evaluate model')
            raise CustomException (str(e),sys)
        
    def save_model(self,model:Sequential):
        try:
            os.makedirs(self.model_output_path, exist_ok=True)  # ensure dir exists
            model_save_path = os.path.join(self.model_output_path, "ann_model.keras")
            model.save(model_save_path)
            logger.info(f"Model saved at {model_save_path}")
        except Exception as e:
            logger.error('Error while saving model')
            raise CustomException(str(e),sys)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info('Starting our Neural Network Training Pipeline')
                logger.info('Logging the training and testing dataset to MLFLOW')
                mlflow.log_artifact(self.train_path,artifact_path='datasets')
                mlflow.log_artifact(self.test_path,artifact_path='datasets')

                X_train , y_train , X_test,y_test = self.load_and_split_data()
                input_dim = X_train.shape[1]
                model = self.create_model_structure(input_size=input_dim)
                self.train_ann(X_train,y_train,model)
                evaluation_metrics = self.evaluate_ann(X_test,y_test,model)
                self.save_model(model)
                logger.info('Logging Params and Metrics to MLFLOW')
                mlflow.log_params({
                    "optimizer":'Adam',
                    "loss":'BinaryCrossentropy',
                    "epochs":20,
                    "batch_size":32,
                    "layers":len(model.layers)
                })
                mlflow.log_metrics({k: float(v) for k, v in evaluation_metrics.items() if k!='confusion_matrix'})

                logger.info('Model Training successfully Completed')
        except Exception as e:
            logger.info('Error while running')
            raise CustomException(str(e),sys)
        
if __name__=='__main__':
    trainer = ANNTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,ANN_OUTPUT_PATH)
    trainer.run()