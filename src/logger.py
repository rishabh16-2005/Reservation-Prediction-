import  logging
import os 
from datetime import datetime

class MLLogger():
    def __init__(self,component_name,log_dir,log_file=None):
        self.component_name = component_name
        self.log_dir = os.path.join(log_dir,component_name)
        os.makedirs(self.log_dir, exist_ok=True)
        if log_file is None:
            time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            log_file = f'ml-log-{time_stamp}.log'
        
        self.log_file = log_file

        self.logger = logging.getLogger(component_name)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(os.path.join(self.log_dir,self.log_file))
            file_handler.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self,message):
        self.logger.info(message)

    def debug(self,message):
        self.logger.debug(message)

    def warning(self,message):
        self.logger.warning(message)

    def error(self,message):
        self.logger.error(message)

    def critical(self,message):
        self.logger.critical(message)

