import sys

def error_mess_detail(error, error_detail:sys):
    _, _, exc_detail = error_detail.exc_info()

    file_name = exc_detail.tb_frame.f_code.co_filename

    error_message = "error is there in filename : [{0}] and line number : [{1}] with error message : [{2}]".format(
        file_name, exc_detail.tb_lineno, str(error))

    return error_message


class CustomExcep(Exception):
    def __init__(self, error_mess, error_detail:sys):
        super().__init__(error_mess)
        self.error_message = error_mess_detail(error_mess, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message