import pandas as pd
import src.processing.util_processing as up
import numpy as np
import ast

filename = "credits.csv"

def raw() :
    return pd.read_csv(up.data_raw_dir+filename)

def get_cast_and_crew_by_id(df,id_film=433):
    row = df.loc[df['id']==id_film]
    cast = ast.literal_eval(row["cast"].values[0])
    crew = ast.literal_eval(row["crew"].values[0])
    return cast,crew

def raw_small(count=1000, save=False, nameifsave="credits_small.csv"):
    dfs = raw()[:count]
    if save:
        dfs.to_csv(up.data_processed_dir+nameifsave)
    return dfs

if __name__=="__main__":
    dfss = raw()
    cast,crew = get_cast_and_crew_by_id(dfss)
    #print(dfss.columns)
    #for i,r in dfss.iterrows():
    #    print(type(r[2]))
    print(cast[0]['cast_id'])
    print(cast[0]['name'])