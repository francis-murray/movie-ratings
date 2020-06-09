import ast

import src.processing.util_processing as up

filename = "keywords.csv"
filename_processed = "keywords_id_3_first_and_idfilm.csv"


def raw():
    df = up.raw(filename)
    return df


def processed():
    return up.processed(filename=filename_processed)


def raw_small(count=1000, save=False, nameifsave="keywords_small.csv", clean=False):
    dfs = up.raw(filename=filename, limit=count)
    if clean:
        dfs = clean(dfs)
    if save:
        dfs.to_csv(up.data_processed_dir + filename)
    return dfs


def get_keywords_of_film(df, id_film=1):
    if df is not None:
        row = df.loc[df["id"] == id_film]
        if not row.empty:
            return row.keywords.values[0]
        else:
            return []
    else:
        return []


def get_keywords_of_film_without_ids(df, id_film=1):
    kw = get_keywords_of_film(df, id_film)
    try:
        if kw:
            return [x['name'] for x in kw]
        else:
            return []
    except:
        return []


def clean(df):
    df['keywords'].fillna('{}', inplace=True)
    df['keywords'] = df['keywords'].apply(lambda x: ast.literal_eval(x))

    def f(kw, x):
        if len(kw) >= x + 1:
            return int(kw[x]['id'])
        else:
            if len(kw) == 0:
                return 0
            else:
                return int(kw[-1]['id'])

    if 'keywordId0' not in df.columns:
        keywordId0 = df['keywords'].apply(lambda x: f(x, 0))
        df['keywordId0'] = keywordId0
    if 'keywordId1' not in df.columns:
        keywordId1 = df['keywords'].apply(lambda x: f(x, 1))
        df['keywordId1'] = keywordId1
    if 'keywordId2' not in df.columns:
        keywordId2 = df['keywords'].apply(lambda x: f(x, 2))
        df['keywordId2'] = keywordId2
    df = df[((df['keywordId0'] != 0) | (df['keywordId1'] != 0) | (df['keywordId2'] != 0))]
    return df


if __name__ == "__main__":
    df = clean(raw())

    df.to_csv(up.data_processed_dir + filename_processed)

    print(df)

    # kw = get_keywords_of_film(df,111109)
    # print(kw)
