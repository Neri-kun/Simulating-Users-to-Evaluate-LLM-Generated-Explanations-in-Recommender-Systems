import re
import pandas as pd

def prepare_user_data(user_df, movies):
    user_id = user_df['userId'].iloc[0]
    user_df = user_df.sort_values(by='timestamp', ascending=False).merge(movies, on='movieId', how='left')
    top_20 = user_df.head(20)
    titles, genres, ratings = top_20['title'].tolist(), top_20['genres'].tolist(), top_20['rating'].tolist()
    return user_id, titles, genres, ratings, top_20

def format_rating_history(titles, ratings, genres):
    return "\n".join(f"{t}: {r} (Genres:{g})" for t, r, g in zip(titles, ratings, genres))

def extract_genre_preferences(df):
    genre_ratings = (
        df.assign(genres=df['genres'].str.split('|'))
          .explode('genres')
          .groupby('genres')['rating']
          .mean()
          .sort_values(ascending=False)
    )
    high = ", ".join(genre_ratings[genre_ratings >= 4].index)
    low = ", ".join(genre_ratings[genre_ratings < 3].index)
    return high, low

def parse_predictions(response):
    pattern = r"-\s*(.+?)\s*:\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, response)
    ratings = [float(r) for _, r in matches]
    titles = [t.strip() for t, _ in matches]
    unmatched = [line for line in response.splitlines() if not re.search(pattern, line)]
    return titles, ratings, unmatched

def parse_response(response):

    RATING_PATTERN = r"-\s*(.+)\s*:\s*(\d+(?:\.\d+)?)"
    raw = re.findall(RATING_PATTERN, response)
    ratings = [float(r) for _, r in raw]
    titles = [t.strip() for t, _ in raw]
    unmatched_parts = [part for part in re.split(RATING_PATTERN, response) if part.strip() not in ('', '\n')]
    unmatched_lines = [i for i, line in enumerate(response.splitlines()) if not re.search(RATING_PATTERN, line)]
    return raw, titles, ratings, unmatched_parts, unmatched_lines

def build_rating_history(titles, ratings, genres):
    return "\n".join(f"{t}: {r} (Genres:{g})" for t, r, g in zip(titles[5:], ratings[5:], genres[5:]))

def format_movies_to_rate(titles, genres):
    return "\n".join(f"{t} (Genre: {g})" for t, g in zip(titles[:5], genres[:5]))

def summarize_genre_preferences(top_20):
    genre_ratings = (
        top_20.tail(15)
              .assign(genres=top_20['genres'].str.split('|'))
              .explode('genres')
              .groupby('genres')['rating']
              .mean()
              .sort_values(ascending=False)
    )
    high = ", ".join(genre_ratings[genre_ratings >= 4].index)
    low = ", ".join(genre_ratings[genre_ratings < 3].index)
    return high, low

def load_and_filter_csv(file_path):
    ratings = pd.read_csv(file_path)
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts == 20].index
    ratings = ratings[ratings['userId'].isin(valid_users)]

    return ratings
