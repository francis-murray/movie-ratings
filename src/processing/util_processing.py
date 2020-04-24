import pandas as pd
import sys
import os
data_processed_dir ="/home/llp0702/Documents/dev/movie-ratings/data/processed/"
data_raw_dir = "/home/llp0702/Documents/dev/movie-ratings/data/raw/"

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