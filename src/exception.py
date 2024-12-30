import sys

# Function to extract detailed error message
def error_mess_detail(error, error_detail: sys):

    # Extract exception details
    _, _, exc_detail = error_detail.exc_info()

    # Retrieve the filename where the error occurred
    file_name = exc_detail.tb_frame.f_code.co_filename

    # Construct an error message with file name, line number, and error description
    error_message = "error is there in filename : [{0}] and line number : [{1}] with error message : [{2}]".format(
        file_name, exc_detail.tb_lineno, str(error)
    )

    return error_message


# Custom exception class to handle errors with detailed information
class CustomExcep(Exception):
    def __init__(self, error_mess, error_detail: sys):
        super().__init__(error_mess)
        # Generate a detailed error message
        self.error_message = error_mess_detail(error_mess, error_detail=error_detail)
    
    def __str__(self):
        """
        Returns the detailed error message as a string.
        """
        return self.error_message
