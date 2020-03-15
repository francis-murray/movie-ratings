import pandas as pd
import src.processing.util_processing as up
import ast

filename = "keywords.csv"

def raw() :
    return pd.read_csv(up.data_raw_dir+filename)

def raw_small(count=1000, save=False, nameifsave="keywords_small.csv"):
    dfs = raw()[:count]
    if save:
        dfs.to_csv(up.data_processed_dir+filename)
    return dfs

def get_keywords_of_film(df,id_film=1):
    if df is not None:
        row = df.loc[df["id"]==id_film]
        if not row.empty:
            return ast.literal_eval(row['keywords'].values[0])
        else:
            return []
    else:
        return []
def get_keywords_of_film_without_ids(df,id_film=1):
    kw = get_keywords_of_film(df,id_film)
    try:
        if kw:
            return [x['name'] for x in kw]
        else:
            return []
    except:
        return []
if __name__=="__main__":
    df = raw()
    kw = get_keywords_of_film_without_ids(df,433)
    print(kw)