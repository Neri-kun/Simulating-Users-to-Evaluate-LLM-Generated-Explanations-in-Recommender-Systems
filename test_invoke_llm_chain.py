import pytest
import asyncio
import pandas as pd
from unittest.mock import MagicMock
from main import _invoke_llm_chain


def load_real_test_data():
    """Load actual movie and rating data from CSV files."""
    try:
        # Load movies
        movies_df = pd.read_csv('movies.csv')

        # Load ratings
        ratings_df = pd.read_csv('ratings.csv')

        # Get a real user with enough ratings for testing
        user_rating_counts = ratings_df['userId'].value_counts()
        valid_users = user_rating_counts[user_rating_counts >= 5].index

        if len(valid_users) == 0:
            raise ValueError("No users with sufficient ratings found")

        test_user_id = valid_users[0]

        # Get this user's rating history
        user_ratings = ratings_df[ratings_df['userId'] == test_user_id].head(10)

        # Convert to the format expected by your function
        rating_history = []
        for _, row in user_ratings.iterrows():
            # Get movie title if available
            movie_title = "Unknown Movie"
            movie_data = movies_df[movies_df['movieId'] == row['movieId']]
            if not movie_data.empty:
                movie_title = movie_data.iloc[0]['title']

            rating_history.append({
                "movie": movie_title,
                "rating": row['rating']
            })

        # Get some movies to rate (movies the user hasn't rated yet)
        rated_movie_ids = user_ratings['movieId'].tolist()
        movies_to_rate_df = movies_df[~movies_df['movieId'].isin(rated_movie_ids)].head(5)
        movies_to_rate = movies_to_rate_df['title'].tolist()

        # Get high and low ratings (for demonstration)
        high_ratings = user_ratings[user_ratings['rating'] >= 4.0].head(3)['movieId']
        low_ratings = user_ratings[user_ratings['rating'] <= 2.0].head(3)['movieId']

        # Convert to movie titles
        high_rating_titles = []
        for movie_id in high_ratings:
            movie_data = movies_df[movies_df['movieId'] == movie_id]
            if not movie_data.empty:
                high_rating_titles.append(movie_data.iloc[0]['title'])

        low_rating_titles = []
        for movie_id in low_ratings:
            movie_data = movies_df[movies_df['movieId'] == movie_id]
            if not movie_data.empty:
                low_rating_titles.append(movie_data.iloc[0]['title'])

        return test_user_id, rating_history, movies_to_rate, high_rating_titles, low_rating_titles

    except FileNotFoundError as e:
        pytest.skip(f"Required CSV files not found: {e}")
    except Exception as e:
        pytest.skip(f"Could not load test data: {e}")


def test_invoke_llm_chain_with_real_data(caplog):
    """Test with actual movie and rating data from CSV files."""
    # Load real data
    user_id, rating_history, movies_to_rate, high_ratings, low_ratings = load_real_test_data()

    print(f"Testing with user {user_id}")
    print(f"Rating history: {len(rating_history)} ratings")
    print(f"Movies to rate: {movies_to_rate}")
    print(f"High rated movies: {high_ratings}")
    print(f"Low rated movies: {low_ratings}")

    # Arrange
    mock_llm_chain = MagicMock()
    mock_llm_chain.invoke.return_value = {"result": "recommendations generated"}

    # Act - run async function synchronously
    result = asyncio.run(_invoke_llm_chain(
        mock_llm_chain,
        user_id,
        rating_history,
        movies_to_rate,
        high_ratings,
        low_ratings
    ))

    # Log the result
    print(f"LLM Chain Result: {result}")

    # Assert
    mock_llm_chain.invoke.assert_called_once()

    # Verify the payload structure
    actual_args, _ = mock_llm_chain.invoke.call_args
    actual_payload = actual_args[0]

    print(f"Actual payload keys: {actual_payload.keys()}")
    print(f"Payload userId: {actual_payload.get('userId')}")
    print(f"Rating history length: {len(actual_payload.get('rating_history', []))}")
    print(f"Movies to rate length: {len(actual_payload.get('movies_to_rate', []))}")

    # Basic assertions about the payload structure
    assert actual_payload['userId'] == user_id
    assert actual_payload['rating_history'] == rating_history
    assert actual_payload['movies_to_rate'] == movies_to_rate
    assert actual_payload['high_ratings'] == high_ratings
    assert actual_payload['low_ratings'] == low_ratings
    assert result == {"result": "recommendations generated"}


def test_invoke_llm_chain_calls_llm_with_correct_args(caplog):
    """Original test with mock data - keep as backup."""
    # Arrange
    mock_llm_chain = MagicMock()
    mock_llm_chain.invoke.return_value = {"result": "ok"}

    user_id = 42
    rating_history = [{"movie": "The Shawshank Redemption", "rating": 5}]
    movies_to_rate = ["Inception", "The Matrix", "Interstellar"]
    high_ratings = ["The Godfather", "Pulp Fiction"]
    low_ratings = ["Twilight", "Transformers"]

    # Act - run async function synchronously
    result = asyncio.run(_invoke_llm_chain(
        mock_llm_chain,
        user_id,
        rating_history,
        movies_to_rate,
        high_ratings,
        low_ratings
    ))

    # Assert
    mock_llm_chain.invoke.assert_called_once()
    expected_payload = {
        "userId": user_id,
        "rating_history": rating_history,
        "movies_to_rate": movies_to_rate,
        "high_ratings": high_ratings,
        "low_ratings": low_ratings
    }
    actual_args, _ = mock_llm_chain.invoke.call_args
    assert actual_args[0] == expected_payload
    assert result == {"result": "ok"}


def test_invoke_llm_chain_handles_exception():
    # Arrange
    mock_llm_chain = MagicMock()
    mock_llm_chain.invoke.side_effect = RuntimeError("LLM error")

    # Act & Assert
    with pytest.raises(RuntimeError, match="LLM error"):
        asyncio.run(_invoke_llm_chain(
            mock_llm_chain,
            user_id=1,
            rating_history=[],
            movies_to_rate=[],
            high_ratings=[],
            low_ratings=[]
        ))