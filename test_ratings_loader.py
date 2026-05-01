# test_ratings_loader.py
import unittest
import pandas as pd
from tempfile import NamedTemporaryFile
from ratings_loader import RatingsLoader


class TestRatingsLoader(unittest.TestCase):
    def _create_temp_csv(self, df):
        tmp = NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        df.to_csv(tmp.name, index=False)
        tmp.close()
        return tmp.name

    def test_filters_users_with_exactly_20_ratings(self):
        df = pd.DataFrame({
            "userId": [1]*20 + [2]*10 + [3]*20,
            "movieId": list(range(50)),
            "rating": [3.5]*50
        })
        path = self._create_temp_csv(df)
        filtered, num_valid = RatingsLoader.load_ratings(path)
        self.assertEqual(num_valid, 2)
        self.assertSetEqual(set(filtered["userId"].unique()), {1, 3})

    def test_no_valid_users_returns_empty_df(self):
        df = pd.DataFrame({
            "userId": [1]*10 + [2]*15,
            "movieId": list(range(25)),
            "rating": [4.0]*25
        })
        path = self._create_temp_csv(df)
        filtered, num_valid = RatingsLoader.load_ratings(path)
        self.assertEqual(num_valid, 0)
        self.assertTrue(filtered.empty)

    def test_raises_error_if_no_userid_column(self):
        df = pd.DataFrame({"movieId": [1, 2, 3], "rating": [4, 3, 5]})
        path = self._create_temp_csv(df)
        with self.assertRaises(ValueError):
            RatingsLoader.load_ratings(path)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            RatingsLoader.load_ratings("nonexistent.csv")
