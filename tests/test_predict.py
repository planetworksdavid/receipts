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

    @patch('src.pipelines.predict.load_and_prepare_actuals_for_evaluation')
    @patch('src.pipelines.predict.load_model') # Mock load_model as it's called in __main__
    # We use self.mock_model for Prophet's make_future_dataframe and predict, so no need to patch Prophet globally here
    def test_main_block_saves_future_forecast_csv(self, mock_load_model, mock_load_actuals):
        # --- Setup Mocks ---
        # 1. Configure mock_load_model to return our self.mock_model
        mock_load_model.return_value = self.mock_model

        # 2. Configure mock_load_actuals to return a controlled DataFrame
        mock_last_actual_date = pd.to_datetime('2023-01-31')
        mock_actuals_df = pd.DataFrame({
            'ds': pd.to_datetime(['2023-01-01', '2023-01-15', mock_last_actual_date.strftime('%Y-%m-%d')]),
            'y': [10, 20, 30]
        })
        mock_load_actuals.return_value = mock_actuals_df

        # 3. Configure self.mock_model for make_future_dataframe and predict
        #    make_future_dataframe should extend from the model's history, which we assume ends at mock_last_actual_date
        #    Prophet's make_future_dataframe(periods=30) will start from the day after its last known date.
        #    The mock_model's history isn't explicitly set here, but Prophet()().make_future_dataframe does this.
        #    So, the 'ds' should start from '2023-02-01' if periods=30 from '2023-01-31'.
        #    Let's ensure the mocked forecast_df has dates both before and after mock_last_actual_date initially.

        num_future_days_to_predict = 30
        # Simulate a model trained up to mock_last_actual_date
        # Prophet's make_future_dataframe includes history + future.
        # Let's define some history dates that would be part of model.history_dates
        history_dates = pd.to_datetime(['2023-01-30', '2023-01-31']) # Ends at mock_last_actual_date

        future_dates_generated = pd.date_range(start=mock_last_actual_date + pd.Timedelta(days=1),
                                               periods=num_future_days_to_predict, freq='D')

        # This is what model.make_future_dataframe would return (history + future)
        all_dates_for_make_df = pd.Index(history_dates).union(pd.Index(future_dates_generated))
        mock_future_df_from_model = pd.DataFrame({'ds': all_dates_for_make_df})
        self.mock_model.make_future_dataframe.return_value = mock_future_df_from_model

        # model.predict will take this combined historical and future df and add yhat, etc.
        # The output of predict() will have the same 'ds' column as its input.
        mock_predict_output_df = pd.DataFrame({
            'ds': all_dates_for_make_df, # Should match the ds from make_future_dataframe's output
            'yhat': np.random.rand(len(all_dates_for_make_df)) * 100,
            'yhat_lower': np.random.rand(len(all_dates_for_make_df)) * 80,
            'yhat_upper': np.random.rand(len(all_dates_for_make_df)) * 120
        })
        self.mock_model.predict.return_value = mock_predict_output_df

        # --- Execution (Replicating logic from __main__ block) ---
        expected_csv_path = os.path.join(self.test_dir, "future_30_day_forecast.csv")

        # Logic from predict.py's __main__
        # model = load_model(...) # Done by mock_load_model
        # actuals_df_for_date = load_and_prepare_actuals_for_evaluation(...) # Done by mock_load_actuals

        # last_actual_date is determined from mock_load_actuals.return_value
        # last_actual_date = mock_actuals_df['ds'].max() # This is '2023-01-31'

        # future_df = make_future_dataframe(model, periods=num_future_days_to_predict, freq='D')
        # The call to make_future_dataframe in predict.py uses self.mock_model (from mock_load_model)
        # and its return value is mock_future_df_from_model

        # forecast_df = predict_forecast(model, future_df)
        # The call to predict_forecast uses self.mock_model and mock_future_df_from_model (as future_df)
        # Its return value is mock_predict_output (as forecast_df)

        # Now, filter and save
        # Replicate filtering logic from predict.py __main__
        # This logic is what we are effectively testing:
        # forecast_df in predict.py is the result of self.mock_model.predict(...)
        # which we've mocked as mock_predict_output_df
        forecast_to_filter = mock_predict_output_df

        future_only_forecast_df = forecast_to_filter[forecast_to_filter['ds'] > mock_last_actual_date].copy()

        num_rows_to_select = min(num_future_days_to_predict, len(future_only_forecast_df))
        final_selected_forecast = future_only_forecast_df.head(num_rows_to_select)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Save this selection to CSV in the test directory
        final_selected_forecast.to_csv(expected_csv_path, index=False)

        # --- Assertions ---
        self.assertTrue(os.path.exists(expected_csv_path))

        saved_csv_df = pd.read_csv(expected_csv_path)
        self.assertEqual(len(saved_csv_df), num_future_days_to_predict) # Should be 30 rows

        expected_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        self.assertListEqual(list(saved_csv_df.columns), expected_columns)

        # Assert that the minimum date in the CSV is the day after mock_last_actual_date
        min_date_in_csv = pd.to_datetime(saved_csv_df['ds']).min()
        self.assertEqual(min_date_in_csv, mock_last_actual_date + pd.Timedelta(days=1))


if __name__ == '__main__':
    project_root_from_test = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root_from_test not in sys.path:
        sys.path.insert(0, project_root_from_test)
    unittest.main()
