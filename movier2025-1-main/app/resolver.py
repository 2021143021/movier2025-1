import pandas as pd


def random_items():
    movies_df = pd.read_csv("data/movies_final.csv")
    movies_df = movies_df.fillna('')
    result_items = movies_df.sample(n=10).to_dict("records")
    return result_items

def random_genres_items(genre):
    movies_df = pd.read_csv("data/movies_final.csv")
    genre_df = movies_df[movies_df['genres'].apply(lambda x: genre in x.lower())]
    genre_df = genre_df.fillna('')
    result_items = genre_df.sample(n=10).to_dict("records")
    return result_items

#지정된 장르를 포함하는 영화중 평점이 높은 5개의 영화를 추천함
def random_genres_items_best(genre):
    movies_df = pd.read_csv("data/movies_final.csv")
    genre_df = movies_df[movies_df['genres'].apply(lambda x: genre in x.lower())]
    genre_df = genre_df.fillna('')
    result_items = genre_df.sample(n=10).to_dict("records")
    return result_items

def random_genres_items_best(genre):
    movies_df = pd.read_csv("data/movies_final.csv")
    movies_df = movies_df.fillna('')

    genre_df = movies_df[
        (movies_df['genres'].apply(lambda x: genre.lower() in x.lower())) &
        (movies_df['rcount'] >= 5)
        ]
    movie_count = len(genre_df)

    result_items = genre_df.head(5).to_dict("records")
    return result_items

