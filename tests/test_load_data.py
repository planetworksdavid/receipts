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
            'transaction_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'transaction_date': [ # Using M/D/YYYY format for strings
                "01/15/2024", "01/15/2024", None, "01/17/2024",
                "05/30/2025", "05/30/2025", "05/31/2025",
                "bad_date_string", "02/29/2023", # Invalid leap day for non-leap year
                "03/01/2024"
            ],
            'total': [500.00, 750.25, 2000.00, 'invalid_total', 15000.00, 8000.00, 7500.00, 600.00, 900.00, 120.75], # Larger totals, one invalid
            'balance_remaining': [0.0, 50.50, 100.0, 20.0, 5000.00, 0.0, 750.00, 60.0, 90.0, 0.0],
            'payment_type': ['Credit Card', 'Cash', 'Debit Card', None, 'Credit Card', 'Online', 'Check', 'Cash', 'Credit Card', 'Debit Card'],
            'voided': [0, 'f', True, '1', 'FALSE', None, 't', 'F', False, "TRUE"], # Diverse voided values
            'time_created': ['2024-01-15 10:00:00', '2024-01-15 10:05:00', '2024-01-16 12:00:00',
                             '2024-01-17 13:00:00', '2025-05-30 14:00:00', '2025-05-30 14:30:00',
                             '2025-05-31 15:00:00', '2024-01-18 10:00:00', '2023-02-29 11:00:00',
                             '2024-03-01 12:00:00'],
            'time_modified': ['2024-01-15 10:00:00', '2024-01-15 11:05:00', '2024-01-16 12:00:00',
                              '2024-01-17 13:05:00', '2025-05-30 14:05:00', '2025-05-30 14:35:00',
                              '2025-05-31 15:10:00', '2024-01-18 10:00:00', '2023-02-29 11:00:00',
                              '2024-03-01 12:05:00'],
            'time_voided': [None, None, '2024-01-16 12:30:00', None, None, None, '2025-05-31 15:15:00',
                            None, None, '2024-03-01 12:10:00']
        }
        self.sample_raw_df = pd.DataFrame(self.sample_raw_data)

        # This df is specifically for testing load_csv_data.
        # It ensures that when saved to CSV, date columns are strings.
        self.csv_ready_df = self.sample_raw_df.copy()
        # For 'transaction_date', it's already strings in M/D/YYYY.
        # For other datetime columns, convert to string and replace NaT if any were pd.NaT initially (though not in this sample).
        for col in ['time_created', 'time_modified', 'time_voided']:
            if col in self.csv_ready_df.columns: # Should always be true here
                 self.csv_ready_df[col] = self.csv_ready_df[col].astype(str).replace('NaT', '')

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.test_dir_tempfile.cleanup()

    def test_load_csv_data_success(self):
        dummy_csv_path = os.path.join(self.test_dir, "test_data.csv")
        self.csv_ready_df.to_csv(dummy_csv_path, index=False)

        loaded_df = load_csv_data(dummy_csv_path)
        self.assertIsNotNone(loaded_df)
        self.assertEqual(len(loaded_df), len(self.csv_ready_df)) # Compare with the df that was saved
        self.assertEqual(len(loaded_df.columns), len(self.csv_ready_df.columns))

        # Assert that 'transaction_date' is parsed correctly with M/D/YYYY format
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded_df['transaction_date']))
        # Check a specific valid date from sample_raw_data (index 0 is "01/15/2024")
        self.assertEqual(loaded_df['transaction_date'].iloc[0], pd.Timestamp('2024-01-15'))
        # Check that "bad_date_string" (index 7) becomes NaT
        self.assertTrue(pd.isna(loaded_df['transaction_date'].iloc[7]))
        # Check that "02/29/2023" (index 8) becomes NaT because it's not a valid date
        self.assertTrue(pd.isna(loaded_df['transaction_date'].iloc[8]))


        # Assert other date columns are also datetime
        other_date_cols = ['time_created', 'time_modified', 'time_voided']
        for col in other_date_cols:
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded_df[col]), f"Column {col} is not datetime")


    def test_load_csv_data_file_not_found(self):
        non_existent_path = os.path.join(self.test_dir, "no_such_file.csv")
        loaded_df = load_csv_data(non_existent_path)
        self.assertIsNone(loaded_df)

    def test_preprocess_data_output_structure(self):
        # Create a DataFrame that simulates the state *after* load_csv_data has run
        # This means 'transaction_date' strings are converted using M/D/YYYY format, others by inference.
        df_post_load_csv = self.sample_raw_df.copy()
        df_post_load_csv['transaction_date'] = pd.to_datetime(df_post_load_csv['transaction_date'], format='%m/%d/%Y', errors='coerce')
        for col in ['time_created', 'time_modified', 'time_voided']:
            df_post_load_csv[col] = pd.to_datetime(df_post_load_csv[col], errors='coerce')
        # Also, 'total' should be numeric for preprocess_data input if 'invalid_total' was there
        df_post_load_csv['total'] = pd.to_numeric(df_post_load_csv['total'], errors='coerce')


        processed_df = preprocess_data(df_post_load_csv) # Pass the correctly formatted DataFrame

        self.assertIn('transaction_day_of_week', processed_df.columns)
        self.assertIn('transaction_month', processed_df.columns)
        self.assertIn('is_voided', processed_df.columns)
        self.assertNotIn('voided', processed_df.columns) # Original 'voided' should be dropped

        self.assertTrue(pd.api.types.is_bool_dtype(processed_df['is_voided']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['total'])) # Should be float after fillna(0.0)
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['balance_remaining'])) # Should be float

        # Check that rows with NaT in transaction_date are dropped.
        # self.sample_raw_df has 3 rows that result in NaT for transaction_date:
        # index 2 (None), index 7 ("bad_date_string"), index 8 ("02/29/2023")
        # So, 10 initial rows - 3 dropped = 7 rows expected.
        self.assertEqual(len(processed_df), 7)


    def test_preprocess_data_missing_values_handled(self):
        # Test data with various missing and specific values to check handling
        test_data_missing = {
            'transaction_date': pd.to_datetime(['01/01/2024', None, '01/03/2024', '01/04/2024'], format='%m/%d/%Y', errors='coerce'),
            'total': [1000, None, 2000, 500], # Test None total
            'balance_remaining': [None, 500, 0, None], # Test None balance
            'payment_type': ['Cash', None, 'Card', 'Check'], # Test None payment_type
            'voided': [0, 'f', True, 0],
            'time_created': pd.to_datetime(['2024-01-01 10:00:00']*4) # Dummy valid times
        }
        df_missing = pd.DataFrame(test_data_missing)

        # 'total' column is already numeric or None, so no pd.to_numeric needed here for test setup

        processed_df = preprocess_data(df_missing.copy())

        # Row with missing transaction_date (index 1) should be dropped
        self.assertEqual(len(processed_df), 3)

        # Check remaining rows (original indices 0, 2, 3)
        # Original index 0: total=1000, balance_remaining=None (should be 0.0)
        self.assertEqual(processed_df.iloc[0]['balance_remaining'], 0.0)
        # Original index 2: total=None (should be 0.0), payment_type=None (should be 'Unknown')
        # This row (original index 1) was dropped.
        # Original index 2 (now index 1 in processed_df): payment_type='Card', total=2000
        self.assertEqual(processed_df.iloc[1]['total'], 2000.0)
        # Original index 3 (now index 2 in processed_df): payment_type=None (should be 'Unknown')
        # This is not how it works, the payment_type was 'Check'.
        # The row where payment_type was None was dropped.
        # Let's re-verify the logic for the row with None payment_type that *is not* dropped.
        # The test_data_missing has 'Cash', None, 'Card', 'Check'. The row with None date is dropped.
        # The remaining are Cash, Card, Check. No 'Unknown' should be generated here.
        # To test 'Unknown' generation, a row *not* dropped for date reasons must have payment_type=None.

        # Let's refine test_data_missing for payment_type
        test_data_payment_type = {
            'transaction_date': pd.to_datetime(['01/01/2024', '01/02/2024'], format='%m/%d/%Y'),
            'total': [1000, 2000],
            'balance_remaining': [0,0],
            'payment_type': ['Cash', None], # Second row has None payment_type
            'voided': [0,0],
            'time_created': pd.to_datetime(['2024-01-01 10:00:00']*2)
        }
        df_payment_type = pd.DataFrame(test_data_payment_type)
        processed_pt_df = preprocess_data(df_payment_type.copy())
        self.assertEqual(processed_pt_df[processed_pt_df['payment_type'] == 'Unknown'].shape[0], 1)
        self.assertEqual(processed_pt_df.iloc[1]['payment_type'], 'Unknown')



    def test_preprocess_data_voided_conversion(self):
        # Test various forms of 'voided' inputs
        voided_test_values = [True, False, 1, 0, 't', 'f', 'True', 'False', 'T', 'F', '1', '0', None, np.nan, 'yes', 'no', ' ', '']
        expected_is_voided = [
            True, False, True, False, True, False, True, False, True, False, True, False,
            False, False, # None, np.nan
            False, False, False, False # 'yes', 'no', ' ', '' (currently map to False by logic)
        ]
        num_test_values = len(voided_test_values)

        df_voided_test = pd.DataFrame({
            'transaction_date': pd.to_datetime([f'01/{i+1:02d}/2024' for i in range(num_test_values)], format='%m/%d/%Y'),
            'total': [1000.0] * num_test_values, # Use larger totals
            'balance_remaining': [0.0] * num_test_values,
            'payment_type': ['Cash'] * num_test_values,
            'voided': voided_test_values,
            'time_created': pd.to_datetime([f'2024-01-{i+1:02d} 10:00:00' for i in range(num_test_values)])
        })

        processed_df = preprocess_data(df_voided_test.copy())

        # Corrected expected_is_voided to match the 18 inputs
        expected_is_voided = [
            True, False, True, False, True, False, True, False, True, False, True, False,
            False, False, # None, np.nan
            False, False, False, False # 'yes', 'no', ' ', ''
        ]

        pd.testing.assert_series_equal(processed_df['is_voided'], pd.Series(expected_is_voided, name='is_voided'), check_dtype=True)


    def test_save_and_load_processed_data(self):
        # Simulate df as it would be after load_csv_data for self.sample_raw_df
        df_post_load_csv = self.sample_raw_df.copy()
        df_post_load_csv['transaction_date'] = pd.to_datetime(df_post_load_csv['transaction_date'], format='%m/%d/%Y', errors='coerce')
        for col in ['time_created', 'time_modified', 'time_voided']:
            df_post_load_csv[col] = pd.to_datetime(df_post_load_csv[col], errors='coerce')
        # Also, handle 'total' if it contains non-numeric strings that load_csv_data wouldn't convert
        df_post_load_csv['total'] = pd.to_numeric(df_post_load_csv['total'], errors='coerce')

        processed_df = preprocess_data(df_post_load_csv) # Use the fully prepared df

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
