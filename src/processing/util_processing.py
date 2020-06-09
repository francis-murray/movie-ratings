import pandas as pd

from config import *

data_raw_dir = ROOT_PATH + '/data/raw/'
data_processed_dir = ROOT_PATH + '/data/processed/'


def raw(filename="", limit=1, all=True, dtype=None):
    if all:
        return pd.read_csv(data_raw_dir + filename, dtype=dtype)
    else:
        return pd.read_csv(data_raw_dir + filename, dtype=dtype)[:limit]


def processed(filename="", limit=1, all=True, dtype=None):
    if all:
        return pd.read_csv(data_processed_dir + filename, dtype=dtype)
    else:
        return pd.read_csv(data_processed_dir + filename, dtype=dtype)[:limit]
