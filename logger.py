from config import *
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _log_unmatched_parts(user_id, response, predicted_ratings, unmatched_parts, missing_predictions):
    """Log detailed information about unmatched parts if significant issues detected."""
    if len(unmatched_parts) > 5:
        logging.debug(f"User {user_id} - Response: {response}")
        logging.debug(f"User {user_id} - Matches: {predicted_ratings}")
        logging.debug(f"User {user_id} - Unmatched parts: {unmatched_parts}")
        logging.debug(f"User {user_id} - Missing predictions: {missing_predictions}")


def _log_successful_processing(user_id, metrics, start_time):
    """Log successful user processing."""
    logging.info(
        f"User {user_id} processed. "
        f"NDCG: {metrics['ndcg']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f} "
        f"in {time.time() - start_time:.2f} sec"
    )

def _build_success_response(metrics, invalid_movies, out_of_range_count,
                            not_allowed_count, attempts, user_id, unmatches, response):
    """Build successful response tuple."""
    return (
        metrics["ndcg"], metrics["mae"], metrics["rmse"],
        0,  # no_of_fallback_predictions
        ["no fallback movie"],  # user_fallback_movies
        out_of_range_count,
        invalid_movies,
        not_allowed_count,
        attempts,
        user_id,
        len(list(unmatches)),
        response
    )

def _log_current_unmatches(user_id, unmatches):
    """Log current unmatches for debugging."""
    if unmatches:
        logging.debug(f"Current unmatches for {user_id}: {list(unmatches)}")