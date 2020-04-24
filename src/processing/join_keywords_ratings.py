import src.processing.process_keywords as kw
import src.processing.process_ratings as rt
import src.processing.util_processing as up

filename="keywords_3_first_id_and_Films_ratings.csv"

def read_raw():
    dfKw = kw.clean(kw.raw())
    dfKw = dfKw.drop(columns=["keywords"])
    dfRt = rt.columns_rating_movieId(rt.raw())
    return dfKw, dfRt
def read_processed():
    dfKw = kw.processed()
    dfRt = rt.processed()

    if 'keywords' in dfKw.columns:
        dfKw = dfKw.drop(columns=["keywords"])
    if 'Unnamed: 0' in dfKw.columns:
        dfKw = dfKw.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in dfRt.columns:
        dfRt = dfRt.drop(columns=['Unnamed: 0'])

    return dfKw, dfRt

def join(dfKw, dfRt):
    print(dfKw.columns)
    print(dfRt.columns)
    result = dfRt.join(dfKw.set_index('id'), on='movieId')
    result['keywordId0'].fillna(0, inplace=True)
    result['keywordId1'].fillna(0, inplace=True)
    result['keywordId2'].fillna(0, inplace=True)
    result = result[((result['keywordId0'] != 0) | (result['keywordId1'] != 0) | (result['keywordId2'] != 0))]
    return result

def save(result,filename):
    result.to_csv(up.data_processed_dir + filename)


def read_raw_and_join_and_save(filename):
    dfKw, dfRt = read_raw()
    result = join(dfKw, dfRt)
    save(result, filename)

def read_processed_and_join_and_save(filename):
    dfKw, dfRt = read_processed()
    result = join(dfKw, dfRt)

    save(result, filename)



def processed():
    result = up.processed(filename)
    result['movieId'] = result['movieId'].astype('int')
    result['number_of_ratings'] = result['number_of_ratings'].astype('int')
    result['rating_mean'] = result['rating_mean'].astype('int')
    result['rating_median'] = result['rating_median'].astype('int')
    result['keywordId0'] = result['keywordId0'].astype('int')
    result['keywordId1'] = result['keywordId1'].astype('int')
    result['keywordId2'] = result['keywordId2'].astype('int')
    return result

if __name__=="__main__":
    read_raw_and_join_and_save(filename)