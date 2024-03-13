# from load import get_image_from_link,save_cvimage
import uuid
from typing import List, Union

import pandas as pd
from tqdm import tqdm

from .extract import COLUMNS, find_filenames, load_dataset
from .load import get_image_from_link, save_cvimage

# import sys
# sys.path.append('..')


class DataTransformer:
    BASE_TO_SAVE = None
    BASE_FILE_PATH = None

    @classmethod
    def extract_and_tranform_image(cls, df: pd.DataFrame) -> List[Union[str, None]]:
        all_created_paths = []
        for idx, row in df.iterrows():
            main, sub, image_url = row.to_list()
            try:
                image = get_image_from_link(image_url)
                image_name = (
                    DataTransformer.BASE_TO_SAVE
                    + "/"
                    + str(main)
                    + "_"
                    + str(sub)
                    + "_"
                    + str(uuid.uuid4())
                )
                extension = "jpg"
                result = save_cvimage(image, image_name, extension)
                if result:
                    all_created_paths.append(image_name + "." + extension)
                else:
                    all_created_paths.append(None)
            except ValueError:
                all_created_paths.append(None)
                continue

        return all_created_paths

    @classmethod
    def transform_all_datasets(cls):
        file_names = find_filenames(DataTransformer.BASE_FILE_PATH)
        dfs = [
            load_dataset(DataTransformer.BASE_FILE_PATH + file_name, COLUMNS)
            for file_name in file_names
        ]
        for idx, df in tqdm(
            enumerate(dfs), total=len(dfs), desc="Transform All Datesets"
        ):
            paths = DataTransformer.extract_and_tranform_image(df)
            if paths:
                df["image_path"] = paths
                df.to_csv(DataTransformer.BASE_TO_SAVE + "/" + file_names[idx] + ".csv")


if __name__ == "__main__":
    pass
