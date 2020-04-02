import pandas as pd
import sys
sys.path.append("../../")
import os
os.chdir('../../')
data_processed_dir = "data/processed/"
data_raw_dir = "data/raw/"
def raw(filename="",limit=1,all=True,dtype=None):
    if all:
        return pd.read_csv(data_raw_dir+filename,dtype=dtype)
    else:
        return pd.read_csv(data_raw_dir+filename,dtype=dtype)[:limit]
def processed(filename="",limit=1, all=True,dtype=None):
    if all:
        return pd.read_csv(data_processed_dir+filename,dtype=dtype)
    else:
        return pd.read_csv(data_processed_dir+filename,dtype=dtype)[:limit]