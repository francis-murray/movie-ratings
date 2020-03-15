import pandas as pd
import src.processing.util_processing as up
import ast

filename = "credits.csv"

def raw() :

    return up.raw(filename)


def raw_small(count=1000, save=False, nameifsave="credits_small.csv"):
    dfs = raw()[:count]
    if save:
        dfs.to_csv(up.data_processed_dir+nameifsave)
    return dfs

def get_cast_and_crew_by_id(df,id_film=433):
    row = df.loc[df['id']==id_film].values[0]
    cast = row[0]
    crew = row[1]
    return cast,crew


def clean(df):
    df['cast'].fillna('{}', inplace=True)
    df['crew'].fillna('{}', inplace=True)
    df['cast'] = df['cast'].apply(lambda x:ast.literal_eval(x))
    df['crew'] = df['crew'].apply(lambda x:ast.literal_eval(x))
    return df

if __name__=="__main__":
    dfss = clean(raw_small())
    cast,crew = get_cast_and_crew_by_id(dfss)
    #print(dfss.columns)
    #for i,r in dfss.iterrows():
    #    print(type(r[2]))
    print(cast[2]['cast_id'])
    print(cast[2]['name'])