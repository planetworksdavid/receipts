import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet # Import Prophet for spec

# Add src to sys.path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipelines.predict import (
    load_model,
    make_future_dataframe,
    predict_forecast,
    evaluate_model,
    load_and_prepare_actuals_for_evaluation
)

class TestPredictPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir_tempfile = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_tempfile.name

        self.mock_model = MagicMock(spec=Prophet) # Use Prophet for spec

        # Configure mock model's methods that are called by the functions under test
        self.future_df_output = pd.DataFrame({
            'ds': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })
        self.mock_model.make_future_dataframe.return_value = self.future_df_output

        self.predict_output = pd.DataFrame({
            'ds': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'yhat': [100, 110, 120],
            'yhat_lower': [90, 100, 110],
            'yhat_upper': [110, 120, 130]
        })
        self.mock_model.predict.return_value = self.predict_output

        self.sample_actuals_data = {
            'ds': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'y': [105, 108]
        }
        self.sample_actuals_df = pd.DataFrame(self.sample_actuals_data)

        # Sample processed data for testing load_and_prepare_actuals_for_evaluation
        self.sample_processed_for_actuals = {
            'transaction_date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-03']),
            'total': [100.0, 5.0, 108.0, 200.0],
            'is_voided': [False, False, False, True], # Last one is voided
             # Other columns that might be present
            'payment_type': ['Card', 'Cash', 'Card', 'Cash'],
            'transaction_day_of_week': [6,6,0,1],
            'transaction_month': [1,1,1,1]
        }
        self.sample_processed_df_for_actuals = pd.DataFrame(self.sample_processed_for_actuals)


    def tearDown(self):
        self.test_dir_tempfile.cleanup()

    def test_load_model_success(self):
        # For this test, save a simple object, as MagicMock itself might not be perfectly stable via joblib
        simple_model_obj = {'name': 'test_prophet_model'}
        dummy_model_path = os.path.join(self.test_dir, "test_model.joblib")
        joblib.dump(simple_model_obj, dummy_model_path)

        loaded = load_model(dummy_model_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['name'], 'test_prophet_model')

    def test_load_model_file_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "no_such_model.joblib")
        loaded = load_model(non_existent_path)
        self.assertIsNone(loaded)

    def test_make_future_dataframe(self):
        periods_to_forecast = 3
        freq_str = 'D'
        # Use a fresh MagicMock for this specific test if needed, or rely on self.mock_model
        test_model_mock = MagicMock()
        test_model_mock.make_future_dataframe.return_value = self.future_df_output # Predefined output

        future_df = make_future_dataframe(test_model_mock, periods=periods_to_forecast, freq=freq_str)

        test_model_mock.make_future_dataframe.assert_called_once_with(periods=periods_to_forecast, freq=freq_str)
        pd.testing.assert_frame_equal(future_df, self.future_df_output)


    def test_predict_forecast(self):
        # Use a fresh MagicMock for this specific test
        test_model_mock = MagicMock()
        test_model_mock.predict.return_value = self.predict_output # Predefined output

        input_future_df = pd.DataFrame({'ds': pd.to_datetime(['2023-01-01', '2023-01-02'])})
        forecast = predict_forecast(test_model_mock, input_future_df)

        test_model_mock.predict.assert_called_once()
        # pd.testing.assert_frame_equal(test_model_mock.predict.call_args[0][0], input_future_df) # Check input to predict
        # Check the DataFrame passed to predict call
        call_args, _ = test_model_mock.predict.call_args
        pd.testing.assert_frame_equal(call_args[0], input_future_df)

        pd.testing.assert_frame_equal(forecast, self.predict_output)


    def test_evaluate_model(self):
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = pd.Series([12.0, 18.0, 33.0])

        expected_mae = np.mean([2.0, 2.0, 3.0]) # |10-12|, |20-18|, |30-33|
        expected_rmse = np.sqrt(np.mean([4.0, 4.0, 9.0])) # (2^2, 2^2, 3^2)

        results = evaluate_model(y_true, y_pred)

        self.assertIsNotNone(results)
        self.assertAlmostEqual(results['mae'], expected_mae, places=4)
        self.assertAlmostEqual(results['rmse'], expected_rmse, places=4)

    def test_evaluate_model_mismatched_lengths_aligned(self):
        # Test alignment logic if implemented, or expected behavior
        y_true = pd.Series([10, 20, 30], index=[0, 1, 2])
        y_pred = pd.Series([12, 18], index=[0, 1]) # Shorter

        # evaluate_model has internal alignment
        results = evaluate_model(y_true, y_pred) # Should align to common index [0,1]

        expected_mae = np.mean([2.0, 2.0])
        expected_rmse = np.sqrt(np.mean([4.0, 4.0]))
        self.assertIsNotNone(results)
        self.assertAlmostEqual(results['mae'], expected_mae, places=4)
        self.assertAlmostEqual(results['rmse'], expected_rmse, places=4)


    def test_evaluate_model_empty_input(self):
        y_true_empty = pd.Series([], dtype=float)
        y_pred_empty = pd.Series([], dtype=float)
        y_true_some = pd.Series([1,2])

        self.assertIsNone(evaluate_model(y_true_empty, y_pred_empty))
        self.assertIsNone(evaluate_model(y_true_some, y_pred_empty))
        self.assertIsNone(evaluate_model(y_true_empty, y_true_some))
        self.assertIsNone(evaluate_model(None, y_true_some))
        self.assertIsNone(evaluate_model(y_true_some, None))

    def test_load_and_prepare_actuals_for_evaluation(self):
        dummy_parquet_path = os.path.join(self.test_dir, "processed_actuals.parquet")
        self.sample_processed_df_for_actuals.to_parquet(dummy_parquet_path, index=False)

        actuals_df = load_and_prepare_actuals_for_evaluation(dummy_parquet_path)

        self.assertIsNotNone(actuals_df)
        self.assertIn('ds', actuals_df.columns)
        self.assertIn('y', actuals_df.columns)

        # Expected:
        # 2023-01-01: 100 + 5 = 105
        # 2023-01-02: 108 (2023-01-03 was voided)
        expected_ds_values = pd.to_datetime(['2023-01-01', '2023-01-02'])
        expected_y_values = pd.Series([105.0, 108.0], name='y')

        pd.testing.assert_series_equal(actuals_df['ds'].reset_index(drop=True), pd.Series(expected_ds_values, name='ds'))
        pd.testing.assert_series_equal(actuals_df['y'].reset_index(drop=True), expected_y_values)
        self.assertEqual(len(actuals_df), 2)


if __name__ == '__main__':
    project_root_from_test = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root_from_test not in sys.path:
        sys.path.insert(0, project_root_from_test)
    unittest.main()
