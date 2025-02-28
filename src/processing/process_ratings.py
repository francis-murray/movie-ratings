from datetime import datetime

import numpy as np
import pandas as pd

import src.processing.util_processing as up

filename = "ratings.csv"
filename_small = "ratings_small.csv"

filename_processed = "ratings_processed.csv"
filename_small_processed = "ratings_small_processed.csv"


def raw():
    return up.raw(filename)


def processed():
    return up.processed(filename_processed)


def raw_small():
    return up.raw(filename_small)


def raw_small_small(nb_rows=100, save=False, nameifsave="ratings_small_small.csv"):
    dfss = raw_small()[:nb_rows]
    if save:
        dfss.to_csv(up.data_processed_dir + nameifsave)
    return dfss


def column_timestamp_to_datetime(df):
    df["timestamp"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x))
    df.columns = ["userId", "movieId", "rating", "datetime"]
    return df


def columns_rating_movieId(df):
    df = df[["rating", "movieId"]]
    df = df.sort_values(["movieId"])
    df_grouped = df.groupby("movieId")
    result = pd.DataFrame(data={
        'movieId': list(df_grouped.groups.keys()),
        'number_of_ratings': np.array(df_grouped["rating"].count()),
        'rating_mean': np.array(df_grouped["rating"].mean()),
        'rating_median': np.array(df_grouped["rating"].median())
    })
    result['rating_median'] = result['rating_median'].astype('int')
    return result


if __name__ == "__main__":
    dfss = raw()
    dfss = columns_rating_movieId(dfss)
    dfss.to_csv(up.data_processed_dir + filename_processed)
    # print(columns_rating_movieId(dfss))
    print(dfss)
