import pandas as pd
import src.processing.util_processing as up
import ast
import src.processing.util_processing as up


filename = "keywords.csv"

def raw() :
    df = up.raw(filename)
    return df

def raw_small(count=1000, save=False, nameifsave="keywords_small.csv", clean=False):
    dfs = up.raw(filename=filename,limit=count)
    if clean:
        dfs=clean(dfs)
    if save:
        dfs.to_csv(up.data_processed_dir+filename)
    return dfs

def get_keywords_of_film(df,id_film=1):
    if df is not None:
        row = df.loc[df["id"]==id_film]
        if not row.empty:
            return row.keywords.values[0]
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

def clean(df):
    df['keywords'].fillna('{}', inplace=True)
    df['keywords'] = df['keywords'].apply(lambda x:ast.literal_eval(x))
    return df

if __name__=="__main__":
    df = clean(raw())
    print(df)
    kw = get_keywords_of_film(df,111109)
    print(kw)

#id