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

        self.sample_processed_data = {
            'transaction_date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03']),
            'total': [100.0, 50.0, 200.0, 30.0, 70.0],
            'is_voided': [False, False, True, False, False],
            # Add other columns that might be present in processed_df but not used by train_prophet_model
            'payment_type': ['Card', 'Cash', 'Card', 'Cash', 'Card'],
            'transaction_day_of_week': [6,6,0,1,1], # Sun, Sun, Mon, Tue, Tue
            'transaction_month': [1,1,1,1,1]
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

        # Verify data preparation logic
        # Expected 'ds' and 'y'
        # Day 1: 100 + 50 = 150 (non-voided)
        # Day 2: (voided, so not included)
        # Day 3: 30 + 70 = 100 (non-voided)
        expected_ds = pd.to_datetime(['2023-01-01', '2023-01-03'])
        expected_y = pd.Series([150.0, 100.0], name='y')
        expected_ds_series = pd.Series(expected_ds, name='ds') # Convert DatetimeIndex to Series

        self.assertEqual(len(fit_df_arg), 2) # After filtering and aggregation
        # fit_df_arg['ds'] already has a reset index from the function being tested
        # The 'name' attribute of the series will be checked by default if both series have it.
        pd.testing.assert_series_equal(fit_df_arg['ds'], expected_ds_series)
        pd.testing.assert_series_equal(fit_df_arg['y'], expected_y)

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
