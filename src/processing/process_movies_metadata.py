import src.processing.util_processing as up
import ast
import datetime as dt
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

def raw():
    return up.raw(filename,dtype=dtype)


def raw_small(count=1000, save=False, nameIfSave="movies_metadata_small.csv"):
    df = up.raw(filename,count,all=False,dtype=dtype)
    if save:
        df.to_csv(up.data_raw_dir+nameIfSave)
    return df

def get_metadata_by_idfilm(df,id_film=0):
    row = df.loc[df['id'] == id_film]
    return row

def clean(df):
    df.drop(['homepage', 'poster_path'], axis=1, inplace=True)
    df.dropna(how='any', inplace=True)
    """df = df[
        (df["adult"].notnull()) &
        (df['belongs_to_collection'].notnull()) &
        (df['budget'].notnull()) &
        (df['genres'].notnull()) &
        (df['original_language'].notnull()) &
        (df['original_title'].notnull()) &
        (df['overview'].notnull()) &
        (df['popularity'].notnull()) &
        ( df['production_companies'].notnull()) &
        ( df['production_countries'].notnull()) &
        (df['release_date'].notnull()) &
        (df['revenue'].notnull()) &
        (df['runtime'].notnull()) &
        (df['spoken_languages'].notnull()) &
        (df['status'].notnull()) &
        ( df['tagline'].notnull()) &
        (df['title'].notnull()) &
        (df['video'].notnull()) &
        ( df['vote_average'].notnull()) &
        (df['vote_count'].notnull())
        ]"""
    """
    df["adult"].fillna(False, inplace=True)
    df['belongs_to_collection'].fillna("{}", inplace=True)
    df['budget'].fillna(df["budget"].mean(), inplace=True)
    df['genres'].fillna('[]', inplace=True)
    df['original_language'].fillna('', inplace=True)
    df['original_title'].fillna('', inplace=True)
    df['overview'].fillna('', inplace=True)
    df['popularity'].fillna(df['popularity'].mean(), inplace=True)
    df['production_companies'].fillna('{}', inplace=True)
    df['production_countries'].fillna('{}', inplace=True)
    df['release_date'].fillna('2000-01-01', inplace=True)
    df['revenue'].fillna(df['revenue'].mean(), inplace=True)
    df['runtime'].fillna(df['runtime'].mean(), inplace=True)
    df['spoken_languages'].fillna('{}', inplace=True)
    df['status'].fillna('', inplace=True)
    df['tagline'].fillna('', inplace=True)
    df['title'].fillna('', inplace=True)
    df['video'].fillna(True, inplace=True)
    df['vote_average'].fillna(df['vote_average'].mean(), inplace=True)
    df['vote_count'].fillna(df['vote_count'].mean(), inplace=True)
    """
    df["adult"] = df["adult"].apply(lambda x:x=='True')
    df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x:ast.literal_eval(x))
    df['genres'] = df['genres'].apply(lambda x:ast.literal_eval(x))
    df['production_companies'] = df['production_companies'].apply(lambda x:ast.literal_eval(x))
    df['production_countries'] = df['production_countries'].apply(lambda x:ast.literal_eval(x))
    df['release_date'] = df['release_date'].apply(lambda x:dt.datetime.strptime(x,'%Y-%m-%d'))
    df['spoken_languages'] = df['spoken_languages'].apply(lambda x:ast.literal_eval(x))
    df["video"] = df['video'].apply(lambda x:x=='True')
    return df

if __name__=="__main__":
    df = clean(raw())
    print(df.columns)

    #print(get_metadata_by_idfilm(df,862))
    print(df.values[0])

#id