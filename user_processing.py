import time
from user_data_preparation import *
from logger import _log_unmatched_parts, _log_successful_processing, _build_success_response,_log_current_unmatches
from validation import _validate_prediction_count, _validate_rating_values, ValidationError, _get_missing_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from config import *
import asyncio
from collections import defaultdict
import os
import threading
from tkinter import ttk, filedialog, messagebox
import numpy as np
from utilities import *

async def process_user_ratings(user_df, movies, llm_chain, max_retries=15, delay=2):
    """Process user ratings with retry logic for LLM chain invocation."""
    start_time_user = time.time()
    user_id, titles, genres, ratings, top_20_rated = prepare_user_data(user_df, movies)

    # Prepare data for LLM
    rating_history = build_rating_history(titles, ratings, genres)
    high_ratings_str, low_ratings_str = summarize_genre_preferences(top_20_rated)
    movies_to_rate = format_movies_to_rate(titles, genres)

    attempts = 0
    missing_predictions = []
    unmatches = set()

    while True:
        try:
            response = await _invoke_llm_chain(llm_chain, user_id, rating_history,
                                               movies_to_rate, high_ratings_str, low_ratings_str)

            print(response)

            parsed_data = parse_response(response)
            predicted_ratings_raw, predicted_titles, predicted_ratings, unmatched_parts, unmatched_lines = parsed_data

            _log_unmatched_parts(user_id, response, predicted_ratings, unmatched_parts, missing_predictions)

            # Validate predictions - pass predicted_titles instead of predicted_ratings_raw
            _validate_prediction_count(predicted_titles, titles, user_id)

            validation_results = _validate_rating_values(predicted_ratings_raw, user_id)
            invalid_rated_movies = validation_results["invalid_rated_movies"]
            no_out_of_range_predicted_ratings = validation_results["out_of_range_count"]
            no_not_allowed_in_range_predicted_ratings = validation_results["not_allowed_count"]

            # Calculate metrics
            metrics = _calculate_evaluation_metrics(ratings[:5], predicted_ratings)

            _log_successful_processing(user_id, metrics, start_time_user)

            return _build_success_response(
                metrics, invalid_rated_movies, no_out_of_range_predicted_ratings,
                no_not_allowed_in_range_predicted_ratings, attempts, user_id, unmatches, response
            )

        except (ValueError, ValidationError) as e:
            attempts += 1
            missing_predictions = _get_missing_predictions(predicted_titles, titles, user_id)
            unmatches.update(missing_predictions)

            logging.warning(f"[Attempt {attempts}] Error processing user {user_id}: {e}")
            _log_current_unmatches(user_id, unmatches)

            await asyncio.sleep(delay)

    return _build_fallback_response(user_id, attempts, unmatches)

async def _invoke_llm_chain(llm_chain, user_id, rating_history, movies_to_rate, high_ratings, low_ratings):
    """Invoke the LLM chain with the provided parameters."""
    return await asyncio.to_thread(llm_chain.invoke, {
        "userId": user_id,
        "rating_history": rating_history,
        "movies_to_rate": movies_to_rate,
        "high_ratings": high_ratings,
        "low_ratings": low_ratings
    })

def _calculate_evaluation_metrics(true_ratings, predicted_ratings):
    """Calculate evaluation metrics (NDCG, MAE, RMSE)."""
    ndcg = ndcg_score([true_ratings], [predicted_ratings])
    mae = mean_absolute_error(true_ratings, predicted_ratings)
    rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))

    return {"ndcg": ndcg, "mae": mae, "rmse": rmse}

async def process_in_batches(new_ratings, movies, llm_chain, batch_size=None):
    """
    Process user ratings in batches asynchronously and collect evaluation metrics.

    Args:
        new_ratings (pd.DataFrame): User ratings DataFrame (must contain 'userId').
        movies (pd.DataFrame): Movies DataFrame used for prediction.
        llm_chain: LLM chain used for rating prediction.
        batch_size (int, optional): Number of users per batch. If None, an optimal size is chosen.

    Returns:
        tuple: (ndcg_scores, mae_scores, rmse_scores, no_of_fallback_predictions, fallback_movies,
                out_of_range_predicted_ratings, invalid_rated_movies,
                no_not_allowed_in_range_predicted_ratings, all_user_attempts, user_ids, unmatches)
    """
    if batch_size is None:
        batch_size = await choose_optimal_batch_size(len(new_ratings))

    logging.info(f"Starting batch processing with batch_size={batch_size}...")

    users = new_ratings['userId'].unique()
    results = defaultdict(list)

    for i in range(0, len(users), batch_size):
        batch_users = users[i:i + batch_size]
        batch_ratings = new_ratings[new_ratings['userId'].isin(batch_users)]

        batch_results = await process_users_in_parallel(
            batch_ratings, movies, llm_chain, max_concurrent=os.cpu_count()
        )

        """Helper to append batch results to the main results dictionary."""
        keys = [
            "ndcg_scores",
            "mae_scores",
            "rmse_scores",
            "no_of_fallback_predictions",
            "fallback_movies",
            "out_of_range_predicted_ratings",
            "invalid_rated_movies",
            "no_not_allowed_in_range_predicted_ratings",
            "all_user_attempts",
            "user_ids",
            "unmatches",
            "responses"
        ]

        for result in batch_results:
            for key, value in zip(keys, result):
                results[key].append(value)


    return tuple(results[key] for key in keys)

async def process_users_in_parallel(new_ratings, movies, llm_chain, max_concurrent=None):
    users = [user_df for _, user_df in new_ratings.groupby('userId')]
    sem = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(user_df):
        async with sem:
            return await process_user_ratings(user_df, movies, llm_chain)

    results = await asyncio.gather(*(process_with_semaphore(user_df) for user_df in users))
    return [r for r in results if r is not None]


def extract_predictions(response):
    pattern = r"-\s*(.+?)\s*:\s*(\d+(?:\.\d+)?)"
    raw = re.findall(pattern, response)
    ratings = [float(r) for _, r in raw]
    titles = [t.strip() for t, _ in raw]
    unmatched_lines = [line for line in response.splitlines() if not re.search(pattern, line)]
    return raw, titles, ratings, unmatched_lines

