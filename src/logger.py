import logging
from datetime import datetime
import os

# Generate a log file name with the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the path for the logs directory and ensure it exists
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)  # Create the directory if it doesn't already exist

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',  # Define the log message format
    filename=LOG_FILE_PATH,  # Log messages will be written to this file
)

# Test the logging setup
if __name__ == '__main__':
    logging.info("Logging has been started")  # Write a test log entry
