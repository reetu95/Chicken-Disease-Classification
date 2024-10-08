import os
import urllib.request as request
import zipfile
import sys
sys.path.append('/Users/reetu/Documents/Personal_Projects/chicken_disease_classification/Chicken-Disease-Classification/src')
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
import urllib.request
import ssl

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            # Disable SSL certificate verification
            ssl._create_default_https_context = ssl._create_unverified_context

            # File download code
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
