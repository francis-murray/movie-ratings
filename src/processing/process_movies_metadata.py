import ast
import math
import datetime as dt
import numpy as np
filename="movies_metadata.csv"
dtype={'adult':str,
      'belongs_to_collection':str,
      'budget':int,
      'genres':str,
      'homepage':str,
      'id':int,
      'imdb_id':str,
      'original_language':str,
      'original_title':str,
      'overview':str,
      'popularity':float,
      'poster_path':str,
      'production_companies':str,
      'production_countries':str,
      'release_date':str,
      'revenue':int,
      'runtime':float,
      'spoken_languages':str,
      'status':str,
      'tagline':str,
      'title':str,
      'video':str,
      'vote_average':float,
      'vote_count':int
    }

import src.processing.util_processing as up

filename = "movies_metadata.csv"
dtype = {'adult': str,
         'belongs_to_collection': str,
         'budget': int,
         'genres': str,
         'homepage': str,
         'id': int,
         'imdb_id': str,
         'original_language': str,
         'original_title': str,
         'overview': str,
         'popularity': float,
         'poster_path': str,
         'production_companies': str,
         'production_countries': str,
         'release_date': str,
         'revenue': int,
         'runtime': float,
         'spoken_languages': str,
         'status': str,
         'tagline': str,
         'title': str,
         'video': str,
         'vote_average': float,
         'vote_count': int
         }


def raw():
    return up.raw(filename, dtype=dtype)


def raw_small(count=1000, save=False, nameIfSave="movies_metadata_small.csv"):
    df = up.raw(filename, count, all=False, dtype=dtype)
    if save:
        df.to_csv(up.data_raw_dir + nameIfSave)
    return df


def get_metadata_by_idfilm(df, id_film=0):
    row = df.loc[df['id'] == id_film]
    return row
def cleanObj(x):
    if  isinstance(x, float) and math.isnan(x):
        return {}
    else :
        return ast.literal_eval(x)
def cleanDateObj(x):
    if  isinstance(x, float) and math.isnan(x):
        return None
    else :
        return dt.datetime.strptime(x,'%Y-%m-%d')
def clean(df):
    df.drop(['homepage', 'poster_path'], axis=1, inplace=True)
    df.dropna(how='all', inplace=True)
    df["adult"] = df["adult"].apply(lambda x:x=='True')
    df['belongs_to_collection'] = df['belongs_to_collection'].apply(cleanObj)
    df['genres'] = df['genres'].apply(cleanObj)
    df['production_companies'] = df['production_companies'].apply(cleanObj)
    df['production_countries'] = df['production_countries'].apply(cleanObj)
    df['release_date'] = df['release_date'].apply(cleanDateObj)
    df['spoken_languages'] = df['spoken_languages'].apply(cleanObj)
    df["video"] = df['video'].apply(lambda x:x=='True')
    df['overview'] = df['overview'].replace(np.nan, '', regex=True)
    df['tagline'] = df['tagline'].replace(np.nan, '', regex=True)
    df['title'] = df['title'].replace(np.nan, '', regex=True)
    df['original_title'] = df['original_title'].replace(np.nan, '', regex=True)
    return df


if __name__ == "__main__":
    df = clean(raw())
    print(df.columns)

    # print(get_metadata_by_idfilm(df,862))
    print(df.values[0])

# id
