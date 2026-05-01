# ratings_loader.py
import pandas as pd

class RatingsLoader:
    """Handles reading and filtering user ratings without UI dependencies."""

    @staticmethod
    def load_ratings(file_path: str, required_ratings: int = 20):
        ratings = pd.read_csv(file_path)

        if 'userId' not in ratings.columns:
            raise ValueError("CSV must contain a 'userId' column")

        user_counts = ratings['userId'].value_counts()
        valid_users = user_counts[user_counts == required_ratings].index
        filtered = ratings[ratings['userId'].isin(valid_users)]

        return filtered, len(valid_users)
