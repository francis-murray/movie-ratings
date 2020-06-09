import src.processing.process_credits as pcredits
import src.processing.process_keywords as pkeywords
import src.processing.process_movies_metadata as pmoviesmd
import src.processing.process_ratings as pratings
import src.processing.util_processing as up

filename = "joined.csv"


def processed(redo=False):
    try:
        if not redo:
            return up.processed(filename)
        else:
            raise Exception
    except:
        df = joined()
        df.to_csv()
        return df


def joined():
    dfCredits = pcredits.clean(pcredits.raw())
    dfKeywords = pkeywords.clean(pkeywords.raw_small())
    dfMoviesMD = pmoviesmd.clean(pmoviesmd.raw_small())
    dfRatings = pratings.raw_small()
    result = dfCredits.join(dfKeywords.set_index('id'), on='id') \
        .join(dfMoviesMD.set_index('id'), on='id') \
        .join(dfRatings.set_index('movieId'), on='id')
    result['vote_average'].fillna(result['vote_average'].mean(), inplace=True)
    result['vote_average'] = result['vote_average'].astype('int')
    return result


# def joined():


if __name__ == '__main__':
    df = processed()
    # print(df.shape)
    print(df.columns)
    print(df)
    print(df.values[1999])
