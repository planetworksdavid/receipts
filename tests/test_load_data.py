import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime

# Add src to sys.path to allow importing from src.pipelines
import sys
# Assuming standard project structure where tests/ is sibling to src/
# So, to import src.pipelines.load_data, we need to add the parent directory of src (project root)
# Or, if running from project root, src itself might need to be identified.
# For `python -m unittest discover tests` or `python tests/test_load_data.py` from root:
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipelines.load_data import load_csv_data, preprocess_data, save_processed_data


class TestLoadDataPipeline(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        self.test_dir_tempfile = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_tempfile.name

        # Sample raw DataFrame mimicking expected CSV structure
        self.sample_raw_data = {
            'transaction_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'transaction_date': ['2023-01-15', '2023-01-16', None, '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-21'],
            'total': [100, 150.50, 200, 'abc', 300, 50.25, None, 120.0],
            'balance_remaining': [0, 50.50, 10.0, 20.0, None, 0, 70.0, 20.0],
            'payment_type': ['Credit Card', 'Cash', 'Debit Card', None, 'Credit Card', 'Online', 'Cash', None],
            'voided': [0, 'f', True, '1', 'FALSE', None, 't', '0'],
            'time_created': ['2023-01-15 10:00:00', '2023-01-16 11:00:00', '2023-01-16 12:00:00',
                             '2023-01-17 13:00:00', '2023-01-18 14:00:00', '2023-01-19 15:00:00',
                             '2023-01-20 16:00:00', '2023-01-21 17:00:00'],
            'time_modified': ['2023-01-15 10:00:00', '2023-01-16 11:05:00', '2023-01-16 12:00:00',
                              '2023-01-17 13:05:00', '2023-01-18 14:00:00', '2023-01-19 15:00:00',
                              '2023-01-20 16:05:00', '2023-01-21 17:05:00'],
            'time_voided': [None, None, '2023-01-16 12:30:00', None, None, None, '2023-01-20 16:10:00', None]
        }
        self.sample_raw_df = pd.DataFrame(self.sample_raw_data)

        # For load_csv_data, ensure date columns are strings before saving to CSV, as they would be in a real CSV
        self.csv_ready_df = self.sample_raw_df.copy()
        for col in ['transaction_date', 'time_created', 'time_modified', 'time_voided']:
            self.csv_ready_df[col] = self.csv_ready_df[col].astype(str).replace('NaT', '') # Replace NaT with empty string for CSV

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.test_dir_tempfile.cleanup()

    def test_load_csv_data_success(self):
        dummy_csv_path = os.path.join(self.test_dir, "test_data.csv")
        self.csv_ready_df.to_csv(dummy_csv_path, index=False)

        loaded_df = load_csv_data(dummy_csv_path)
        self.assertIsNotNone(loaded_df)
        self.assertEqual(len(loaded_df), len(self.sample_raw_df))
        self.assertEqual(len(loaded_df.columns), len(self.sample_raw_df.columns))

        date_columns = ['transaction_date', 'time_created', 'time_modified', 'time_voided']
        for col in date_columns:
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded_df[col]), f"Column {col} is not datetime")

        # Check a specific conversion for a non-None date
        self.assertEqual(loaded_df['transaction_date'].iloc[0], pd.Timestamp('2023-01-15'))


    def test_load_csv_data_file_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "no_such_file.csv")
        loaded_df = load_csv_data(non_existent_path)
        self.assertIsNone(loaded_df)

    def test_preprocess_data_output_structure(self):
        # load_csv_data converts dates, so we simulate that before passing to preprocess
        df_for_preprocessing = self.sample_raw_df.copy()
        date_cols_for_sample = ['transaction_date', 'time_created', 'time_modified', 'time_voided']
        for col in date_cols_for_sample:
            df_for_preprocessing[col] = pd.to_datetime(df_for_preprocessing[col], errors='coerce')

        processed_df = preprocess_data(df_for_preprocessing)

        self.assertIn('transaction_day_of_week', processed_df.columns)
        self.assertIn('transaction_month', processed_df.columns)
        self.assertIn('is_voided', processed_df.columns)
        self.assertNotIn('voided', processed_df.columns) # Original 'voided' should be dropped

        self.assertTrue(pd.api.types.is_bool_dtype(processed_df['is_voided']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['total']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['balance_remaining']))

        # Check that rows with NaT in transaction_date are dropped
        # Original sample_raw_df has one NaT after pd.to_datetime(errors='coerce')
        self.assertEqual(len(processed_df), len(self.sample_raw_df) - 1)


    def test_preprocess_data_missing_values_handled(self):
        test_data_missing = {
            'transaction_date': [None, '2023-01-16', '2023-01-17'],
            'total': [100, None, 200],
            'balance_remaining': [None, 50, None],
            'payment_type': ['Cash', None, 'Card'],
            'voided': [0, 1, 0], # Needs to be present for the function
            'time_created': ['2023-01-15 10:00:00', '2023-01-16 11:00:00', '2023-01-17 12:00:00'] # Required non-date features
        }
        df_missing = pd.DataFrame(test_data_missing)
        # Convert date columns like load_csv_data would
        df_missing['transaction_date'] = pd.to_datetime(df_missing['transaction_date'], errors='coerce')
        df_missing['time_created'] = pd.to_datetime(df_missing['time_created'], errors='coerce')


        processed_df = preprocess_data(df_missing.copy())

        # Row with missing transaction_date should be dropped
        self.assertEqual(len(processed_df), 2)

        # Check remaining rows (original index 1 and 2, now 0 and 1 in processed_df)
        # Original index 1 had None for total
        self.assertEqual(processed_df.loc[processed_df['payment_type'] == 'Unknown', 'total'].iloc[0], 0)
        # Original index 0 had None for balance_remaining (but this row is dropped)
        # Original index 2 had None for balance_remaining
        self.assertEqual(processed_df.loc[processed_df['payment_type'] == 'Card', 'balance_remaining'].iloc[0], 0)

        self.assertTrue('Unknown' in processed_df['payment_type'].values)


    def test_preprocess_data_voided_conversion(self):
        voided_test_values = [True, False, 1, 0, 't', 'f', 'True', 'False', 'T', 'F', '1', '0', None, np.nan]
        # Ensure other critical columns are present and valid for the same number of rows
        num_test_values = len(voided_test_values)
        df_voided_test = pd.DataFrame({
            'transaction_date': pd.to_datetime([f'2023-01-{i+1:02d}' for i in range(num_test_values)]),
            'total': [100] * num_test_values,
            'balance_remaining': [0] * num_test_values,
            'payment_type': ['Cash'] * num_test_values,
            'voided': voided_test_values,
            'time_created': pd.to_datetime([f'2023-01-{i+1:02d} 10:00:00' for i in range(num_test_values)])
        })

        processed_df = preprocess_data(df_voided_test.copy())

        expected_is_voided = [
            True, False, True, False, True, False, True, False, True, False, True, False, False, False # None and np.nan become False
        ]

        pd.testing.assert_series_equal(processed_df['is_voided'], pd.Series(expected_is_voided, name='is_voided'), check_dtype=True)


    def test_save_and_load_processed_data(self):
        # Simulate df as it would be after load_csv_data
        df_for_preprocessing = self.sample_raw_df.copy()
        date_cols_for_sample = ['transaction_date', 'time_created', 'time_modified', 'time_voided']
        for col in date_cols_for_sample:
            df_for_preprocessing[col] = pd.to_datetime(df_for_preprocessing[col], errors='coerce')

        processed_df = preprocess_data(df_for_preprocessing)

        # Ensure there's some data to save (preprocess_data drops one row with NaT date)
        self.assertTrue(not processed_df.empty)

        test_parquet_path = os.path.join(self.test_dir, "processed_data.parquet")
        save_processed_data(processed_df, test_parquet_path)

        self.assertTrue(os.path.exists(test_parquet_path))

        loaded_df_parquet = pd.read_parquet(test_parquet_path)

        # Pandas testing can be strict about dtypes (e.g. int32 vs int64)
        # preprocess_data results in int32 for dayofweek/month. Parquet might preserve or alter this.
        # For robust comparison, especially if only values matter and some dtype variations are acceptable:

        # Reset index of processed_df because it might be non-contiguous after dropping rows,
        # while read_parquet will produce a default RangeIndex if index=False was used during save.
        processed_df_reset = processed_df.reset_index(drop=True)

        pd.testing.assert_frame_equal(processed_df_reset, loaded_df_parquet, check_dtype=True)


if __name__ == '__main__':
    # This setup of sys.path is primarily for running the test script directly.
    # If using `python -m unittest discover tests` from project root, it might not be necessary
    # as the project root is often implicitly part of Python's path resolution.
    # However, being explicit can help in various execution contexts.

    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root_from_test_file = os.path.abspath(os.path.join(current_dir, '..'))
    # src_path = os.path.join(project_root_from_test_file, 'src')

    # if src_path not in sys.path:
    #    sys.path.insert(0, src_path)
    # if project_root_from_test_file not in sys.path: # To resolve src.pipelines...
    #    sys.path.insert(0, project_root_from_test_file)
    # The sys.path modification at the top of the file is generally preferred.

    unittest.main()
