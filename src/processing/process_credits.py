import src.processing.util_processing as up
import ast
import pandas as pd
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
    df.dropna(how='any', inplace=True)
    df['cast'] = df['cast'].apply(lambda x:ast.literal_eval(x))
    df['crew'] = df['crew'].apply(lambda x:ast.literal_eval(x))
    return df

def get_all_cast(df=None):
    try:
        if df is None:
            cast = pd.read_csv(up.data_processed_dir+"actors.csv")
            if 'Unnamed: 0' in cast.columns:
                cast =cast.drop(columns=['Unnamed: 0'])
            print(cast)
            print(len(cast))
            print(cast.values[0])
            return cast
        else : raise Exception
    except:
        if df is None:
            df = clean(raw())
        df = df['cast']
        cast = []

        def add_all(x):
            for i in x:
                cast.append({'id': i['id'], 'gender': i['gender'], 'name': i['name']})

        df.apply(add_all)

        cast = [dict(t) for t in {tuple(d.items()) for d in cast}]
        print(cast)
        print(len(cast))
        print(cast[0])
        cast = pd.DataFrame(cast)
        cast.to_csv(up.data_processed_dir + "actors.csv")
        return cast


if __name__=="__main__":
    #dfss = clean(raw())
    get_all_cast()
    print("salut")
#id