import traceback
import sys 

class CustomException(Exception):
    def __init__(self, error_message , error_detail:sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message,error_detail)
    
    @staticmethod
    def get_detailed_error_message(error_message,error_details:sys):

        _ , _ , exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no  = exc_tb.tb_lineno

        return f"Error occured in {file_name} , line {line_no} : {error_message}"
    
    def __str__(self):
        return self.error_message