from os import listdir
from typing import List

import pandas as pd
from tqdm import tqdm

COLUMNS = ["main_category", "sub_category", "image"]


def load_dataset(path: str, cols: List[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=cols)


def process_all_csv(
    source: str, file_names: List[str], cols: str, destination_base: str
):

    dfs: List[pd.DataFrame] = [load_dataset(source + file, cols) for file in file_names]
    for df, file_name in tqdm(
        zip(dfs, file_names), desc="Processing Raw Files", total=len(file_names)
    ):
        df.to_csv(path_or_buf=destination_base + file_name)


def find_filenames(path_to_dir, suffix=".csv") -> List[str]:
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


if __name__ == "__main__":
    base_path = "data/raw/v1/"
    destionation = "data/processed/v1/"
    # process_all_csvs([test_path],['Air Conditioners'],COLUMNS,destination_base=destionation)
    file_names = find_filenames(base_path)
    process_all_csv(
        source=base_path,
        file_names=file_names,
        cols=COLUMNS,
        destination_base=destionation,
    )
