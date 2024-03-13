from scripts.transform import DataTransformer

BASE_TO_SAVE = "/home/ahmedelsayed/Projects/Product-Categorization/data/present/modelling/v1/images"
BASE_FILE_PATH = "/home/ahmedelsayed/Projects/Product-Categorization/data/processed/v1/"
DataTransformer.BASE_FILE_PATH = BASE_FILE_PATH
DataTransformer.BASE_TO_SAVE = BASE_TO_SAVE
DataTransformer.transform_all_datasets()
