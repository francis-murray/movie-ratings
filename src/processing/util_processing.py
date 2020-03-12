import pandas as pd

data_processed_dir = "../../data/processed/"
data_raw_dir = "../../data/raw/"

def raw(filename="",limit=1,all=True):
    if all:
        return pd.read_csv(data_raw_dir+filename)
    else:
        return pd.read_csv(data_raw_dir+filename)[:limit]
def processed(filename="",limit=1, all=True):
    if all:
        return pd.read_csv(data_processed_dir+filename)
    else:
        return pd.read_csv(data_processed_dir+filename)[:limit]