ALLOWED_RATINGS = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}

import logging
# de lucrate la asta (probleme cu len(titles) != len(missing_titles) + len(predicted_titles)
def _validate_prediction_count(predicted_titles, expected_titles, user_id):
    """Validate that we have sufficient predictions."""
    MIN_REQUIRED_PREDICTIONS = 5

    if len(predicted_titles) < MIN_REQUIRED_PREDICTIONS:
        expected_movies = expected_titles[:MIN_REQUIRED_PREDICTIONS]
        predicted_titles_set = set(predicted_titles)
        missing_predictions = [title for title in expected_movies if title not in predicted_titles_set]

        raise ValidationError(
            f"Too few predicted ratings: {len(predicted_titles)} for {user_id}. "
            f"Missing movies: {missing_predictions}"
        )

def validate_predictions(predicted_ratings_raw):
    invalid_movies = []
    out_of_range = 0
    not_allowed = 0

    for title, rating_str in predicted_ratings_raw:
        rating = float(rating_str)
        if not 0.5 <= rating <= 5.0:
            invalid_movies.append((title.strip(), rating))
            out_of_range += 1
            raise ValueError(f"Out of range predicted rating: {rating}")
        elif rating not in ALLOWED_RATINGS:
            not_allowed += 1
            raise ValueError(f"Not allowed predicted rating: {rating}")

    return invalid_movies, out_of_range, not_allowed


def _validate_rating_values(predicted_ratings_raw, user_id):
    """Validate that all ratings are within allowed range and values."""
    invalid_rated_movies = []
    out_of_range_count = 0
    not_allowed_count = 0

    for title, rating_str in predicted_ratings_raw:
        rating = float(rating_str)

        if not 0.5 <= rating <= 5.0:
            invalid_rated_movies.append((title.strip(), rating))
            out_of_range_count += 1
            raise ValidationError(f"Out of range predicted rating: {rating} in movie '{title}' for '{user_id}'")

        elif rating not in ALLOWED_RATINGS:
            not_allowed_count += 1
            raise ValidationError(f"Not allowed predicted rating within range: {rating} in movie '{title}' for '{user_id}'")

    return {
        "invalid_rated_movies": invalid_rated_movies,
        "out_of_range_count": out_of_range_count,
        "not_allowed_count": not_allowed_count
    }

def _get_missing_predictions(predicted_titles, expected_titles, user_id):
    """
    Return a list of missing predictions if fewer than the minimum required
    number of predictions are provided.
    """
    MIN_REQUIRED_PREDICTIONS = 5

    # Limit expected titles to the number of movies we actually care about
    expected_movies = expected_titles[:MIN_REQUIRED_PREDICTIONS]
    predicted_titles_set = set(predicted_titles)

    # Compute which expected movies are missing
    missing_predictions = [title for title in expected_movies if title not in predicted_titles_set]

    if len(predicted_titles) < MIN_REQUIRED_PREDICTIONS:
        logging.warning(
            f"[User {user_id}] Too few predicted ratings "
            f"({len(predicted_titles)}/{MIN_REQUIRED_PREDICTIONS}). "
            f"Missing: {missing_predictions}"
        )

    return missing_predictions
# Custom exception for validation errors
class ValidationError(ValueError):
    """Custom exception for validation-related errors."""
    pass
