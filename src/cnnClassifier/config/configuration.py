import sys
import os

# Clear out any previous paths in the notebook by restarting kernel

# Append the correct path to the 'src' directory inside the nested folder
sys.path.append('/Users/reetu/Documents/Personal_Projects/chicken_disease_classification/Chicken-Disease-Classification/src')

# Print the final sys.path to verify the path is added only once
# print("Final sys.path:", sys.path)

# Try importing the modules again
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config