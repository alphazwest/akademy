import os.path
from unittest import TestCase
from akademy.common.project_paths import SPY_DATA
import pandas as pd


class TestData(TestCase):
    """
    Test suite to ensure all expected data is available to the project.
    """
    def test_spy_csv(self):
        """
        Tests that the SPY.csv file is available as expected.
        """
        self.assertTrue(os.path.exists(SPY_DATA))
        df = pd.read_csv(SPY_DATA)
        self.assertTrue(all([

            # test all expected columns required for training are present
            [
                x in df.columns
                for x in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            ],

            # test the expected length
            len(df) > 7454

        ]))
