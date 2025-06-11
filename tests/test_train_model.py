import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import joblib
import os
import tempfile
from datetime import datetime
import sys # Import the sys module

# Add src to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipelines.train_model import load_processed_data, train_prophet_model, save_model
# from prophet import Prophet # We will mock Prophet, so direct import not strictly needed for test logic

class TestTrainModelPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir_tempfile = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_tempfile.name

        # Sample processed DataFrame with recent/future dates and larger totals
        # Dates are already datetime objects, as they would be after load_data.py
        self.sample_processed_data = {
            'transaction_date': pd.to_datetime([
                "01/15/2024", "01/15/2024", # Duplicate date
                "01/16/2024",               # Single transaction
                "05/30/2025", "05/30/2025", # Future duplicate date
                "05/31/2025"                # Future single transaction
            ], format="%m/%d/%Y"),
            'total': [500.0, 750.25, 1200.0, 15000.0, 8000.0, 7500.0], # Larger totals
            'is_voided': [False, False, True, False, False, False], # One voided transaction
            'payment_type': ['Card', 'Cash', 'Card', 'Cash', 'Card', 'Online'],
            'transaction_day_of_week': pd.to_datetime([
                "01/15/2024", "01/15/2024", "01/16/2024",
                "05/30/2025", "05/30/2025", "05/31/2025"
            ], format="%m/%d/%Y").dayofweek,
            'transaction_month': pd.to_datetime([
                "01/15/2024", "01/15/2024", "01/16/2024",
                "05/30/2025", "05/30/2025", "05/31/2025"
            ], format="%m/%d/%Y").month
        }
        self.sample_processed_df = pd.DataFrame(self.sample_processed_data)

    def tearDown(self):
        self.test_dir_tempfile.cleanup()

    def test_load_processed_data_success(self):
        dummy_parquet_path = os.path.join(self.test_dir, "processed_data.parquet")
        self.sample_processed_df.to_parquet(dummy_parquet_path, index=False)

        loaded_df = load_processed_data(dummy_parquet_path)
        self.assertIsNotNone(loaded_df)
        pd.testing.assert_frame_equal(loaded_df, self.sample_processed_df)

    def test_load_processed_data_file_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "no_such_file.parquet")
        loaded_df = load_processed_data(non_existent_path)
        self.assertIsNone(loaded_df)

    @patch('src.pipelines.train_model.Prophet')
    def test_train_prophet_model(self, MockProphet):
        mock_model_instance = MockProphet.return_value
        mock_model_instance.fit.return_value = None  # Prophet's fit method usually returns self (the model instance)

        returned_model = train_prophet_model(self.sample_processed_df.copy()) # Pass a copy

        MockProphet.assert_called_once()  # Check if Prophet() was instantiated

        # Check if fit was called. The argument to fit is a DataFrame.
        self.assertTrue(mock_model_instance.fit.called)
        args, kwargs = mock_model_instance.fit.call_args
        self.assertTrue(len(args) > 0, "fit called with no positional arguments")
        fit_df_arg = args[0] # The DataFrame passed to fit

        self.assertIsInstance(fit_df_arg, pd.DataFrame)
        self.assertIn('ds', fit_df_arg.columns)
        self.assertIn('y', fit_df_arg.columns)

        # Verify data preparation logic based on updated self.sample_processed_df
        # Original data:
        # 01/15/2024: 500.0 (False) + 750.25 (False) = 1250.25
        # 01/16/2024: 1200.0 (True) -> voided, so not included
        # 05/30/2025: 15000.0 (False) + 8000.0 (False) = 23000.0
        # 05/31/2025: 7500.0 (False) = 7500.0

        expected_dates_str = ["01/15/2024", "05/30/2025", "05/31/2025"]
        expected_ds = pd.to_datetime(expected_dates_str, format="%m/%d/%Y")
        expected_y_values = [1250.25, 23000.0, 7500.0]

        expected_y = pd.Series(expected_y_values, name='y')
        expected_ds_series = pd.Series(expected_ds, name='ds')

        self.assertEqual(len(fit_df_arg), 3) # After filtering voided and aggregation

        # Ensure ds and y columns in fit_df_arg match expected values
        # The function under test already does reset_index()
        pd.testing.assert_series_equal(fit_df_arg['ds'], expected_ds_series, check_dtype=False) # Allow date dtype variations if any
        pd.testing.assert_series_equal(fit_df_arg['y'], expected_y, check_dtype=False) # Allow float dtype variations

        self.assertIs(returned_model, mock_model_instance)


    def test_train_prophet_model_empty_input(self):
        empty_df = pd.DataFrame()
        result = train_prophet_model(empty_df)
        self.assertIsNone(result, "Should return None for empty DataFrame input")

    def test_train_prophet_model_no_non_voided(self):
        df_all_voided = self.sample_processed_df.copy()
        df_all_voided['is_voided'] = True
        result = train_prophet_model(df_all_voided)
        self.assertIsNone(result, "Should return None if all transactions are voided")


    def test_save_model(self):
        # Use a simple, pickleable object instead of MagicMock directly for joblib
        dummy_model_dict = {'some_attribute': "test_value", 'type': 'test_model'}

        model_path = os.path.join(self.test_dir, "test_model.joblib")
        save_model(dummy_model_dict, model_path)

        self.assertTrue(os.path.exists(model_path))

        loaded_model = joblib.load(model_path)
        self.assertIsInstance(loaded_model, dict)
        self.assertEqual(loaded_model['some_attribute'], "test_value")
        self.assertEqual(loaded_model['type'], "test_model")


if __name__ == '__main__':
    project_root_from_test = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root_from_test not in sys.path:
        sys.path.insert(0, project_root_from_test)
    unittest.main()
