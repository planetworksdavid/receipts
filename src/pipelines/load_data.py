"""
Handles loading raw payment data from CSV, preprocessing it, and saving the processed data.

This module provides functionalities to:
- Load data from a CSV file.
- Convert specified string columns to datetime objects.
- Perform data cleaning and preprocessing, including:
    - Handling missing values.
    - Engineering new features (e.g., day of week, month from transaction date).
    - Standardizing data types (e.g., ensuring 'is_voided' is boolean).
- Save the processed DataFrame to a Parquet file for efficient downstream use.
"""
import pandas as pd
import numpy as np # Used for np.nan, though often pandas handles this implicitly.

def load_csv_data(file_path: str) -> pd.DataFrame | None:
    """
    Reads a CSV file into a pandas DataFrame and converts specified columns to datetime objects.

    The function targets 'transaction_date', 'time_created', 'time_modified',
    and 'time_voided' columns for datetime conversion. If a column is not present,
    it is skipped. Errors encountered during date parsing are coerced, resulting in NaT
    (Not a Time) values for problematic entries.

    :param file_path: Path to the CSV file.
    :type file_path: str
    :return: A pandas DataFrame with relevant date columns converted to datetime types.
             Returns None if the file is not found or another exception occurs during loading.
    :rtype: pd.DataFrame or None
    """
    try:
        df = pd.read_csv(file_path)
        date_columns = ['transaction_date', 'time_created', 'time_modified', 'time_voided']
        if 'transaction_date' in df.columns:
            # Explicitly use format for 'transaction_date'
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%m/%d/%Y', errors='coerce')

        # For other date/time columns, pandas' default inference is usually fine
        other_date_cols = ['time_created', 'time_modified', 'time_voided']
        for col in other_date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e: # Catch other potential exceptions like pd.errors.EmptyDataError
        print(f"Error loading or converting data from {file_path}: {e}")
        return None

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses an input DataFrame containing transaction data.

    Key preprocessing steps include:
    1.  **Missing Value Imputation**:
        - Fills missing 'total' and 'balance_remaining' with 0 after numeric conversion.
        - Fills missing 'payment_type' with the string 'Unknown'.
        - Drops rows where 'transaction_date' is missing or invalid (NaT).
    2.  **Feature Engineering**:
        - Creates 'transaction_day_of_week' from 'transaction_date' (Monday=0, Sunday=6).
        - Creates 'transaction_month' from 'transaction_date'.
        - Derives a boolean 'is_voided' column from the 'voided' column, interpreting
          various common representations of true/false.
    3.  **Data Type Standardization**:
        - Ensures 'total' and 'balance_remaining' are float type.
        - Ensures 'is_voided' is boolean type.
    4.  **Column Cleanup**:
        - Removes the original 'voided' column after 'is_voided' is created.

    The function operates on a copy of the input DataFrame to avoid unintended side effects.

    :param df: Input pandas DataFrame. Expected to include columns like 'transaction_date',
               'total', 'balance_remaining', 'payment_type', and 'voided'.
    :type df: pd.DataFrame
    :return: A new pandas DataFrame with preprocessing applied. If essential columns
             like 'transaction_date' are entirely missing or all rows are dropped,
             it may return a modified or empty DataFrame with warnings printed.
    :rtype: pd.DataFrame
    """
    print("Starting preprocessing...")
    processed_df = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # 1. Handle Missing Values
    # Convert 'total' and 'balance_remaining' to numeric, coercing errors (non-numeric values become NaN), then fill NaN with 0.
    processed_df['total'] = pd.to_numeric(processed_df['total'], errors='coerce').fillna(0.0)
    processed_df['balance_remaining'] = pd.to_numeric(processed_df['balance_remaining'], errors='coerce').fillna(0.0)
    processed_df['payment_type'] = processed_df['payment_type'].fillna('Unknown')

    # Handle 'transaction_date': it's critical. If missing or invalid (NaT), drop the row.
    if 'transaction_date' in processed_df.columns:
        initial_rows = len(processed_df)
        # Drop rows where 'transaction_date' is NaT (Not a Time)
        processed_df.dropna(subset=['transaction_date'], inplace=True)
        rows_dropped = initial_rows - len(processed_df)
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows due to missing or invalid 'transaction_date'.")
    else:
        # If 'transaction_date' column doesn't exist, further processing is problematic.
        print("Warning: 'transaction_date' column not found. Essential for date-based features and analysis.")
        return processed_df # Return the DataFrame as is, or consider raising an error.

    # If all rows were dropped (e.g., no valid transaction_date), return the empty DataFrame.
    if processed_df.empty:
        print("Warning: DataFrame is empty after handling missing 'transaction_date'.")
        return processed_df

    # 2. Feature Engineering
    # Safeguard: Ensure 'transaction_date' is actually datetime dtype before using .dt accessor.
    # This should ideally be handled by load_csv_data, but this makes preprocess_data more robust.
    if not pd.api.types.is_datetime64_any_dtype(processed_df['transaction_date']):
         processed_df['transaction_date'] = pd.to_datetime(processed_df['transaction_date'], errors='coerce')
         # Re-validate and drop rows if new NaTs were introduced
         initial_rows = len(processed_df)
         processed_df.dropna(subset=['transaction_date'], inplace=True)
         rows_dropped_after_coercion = initial_rows - len(processed_df)
         if rows_dropped_after_coercion > 0:
            print(f"Dropped {rows_dropped_after_coercion} additional rows due to 'transaction_date' coercion errors during preprocessing safeguard.")
         if processed_df.empty: # Check if DataFrame became empty after this additional drop
            print("Warning: DataFrame became empty after 'transaction_date' coercion safeguard.")
            return processed_df

    # Create new date-based features
    processed_df['transaction_day_of_week'] = processed_df['transaction_date'].dt.dayofweek # Monday=0, Sunday=6
    processed_df['transaction_month'] = processed_df['transaction_date'].dt.month

    # Create 'is_voided' (boolean) from 'voided' column (which can have mixed types)
    if 'voided' in processed_df.columns:
        # Define a set of common string/numeric representations of True
        true_values_set = {True, 'True', 'true', 't', 'T', 1, '1'}
        # Apply conversion: if value is in true_values_set or its lowercase string version is, map to True. Else False.
        processed_df['is_voided'] = processed_df['voided'].apply(
            lambda x: x in true_values_set or \
                      (isinstance(x, str) and x.lower() in true_values_set)
        )
    else:
        print("Warning: 'voided' column not found. 'is_voided' feature will be created as all False.")
        processed_df['is_voided'] = False # Default to False if original 'voided' column is missing

    # 3. Data Type Conversion (Final check and enforcement)
    # Ensures 'total' and 'balance_remaining' are float, even if all values were integers after fillna(0).
    processed_df['total'] = processed_df['total'].astype(float)
    processed_df['balance_remaining'] = processed_df['balance_remaining'].astype(float)
    processed_df['is_voided'] = processed_df['is_voided'].astype(bool)

    # Drop the original 'voided' column as 'is_voided' is the standardized boolean representation.
    if 'voided' in processed_df.columns:
        processed_df.drop(columns=['voided'], inplace=True)
        # print("Dropped original 'voided' column.") # Optional: uncomment for verbose logging

    print("Preprocessing complete.")
    return processed_df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the given DataFrame to a Parquet file.

    This function uses the 'pyarrow' engine for writing the Parquet file
    and does not include the DataFrame's index in the output file.

    :param df: The pandas DataFrame to be saved.
    :type df: pd.DataFrame
    :param output_path: The file path where the Parquet file should be saved.
    :type output_path: str
    :return: None
    :rtype: None
    :raises Exception: Prints an error message to the console if any exception occurs
                       during the file saving process (e.g., invalid path, permissions error).
    """
    try:
        # Save to Parquet format using pyarrow engine, do not write index
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Processed data successfully saved to {output_path}")
    except Exception as e:
        # Broad exception catch for any issues during file saving (e.g., path invalid, permissions)
        print(f"Error saving data to Parquet file at {output_path}: {e}")


if __name__ == "__main__":
    # This main block serves as a demonstration or a simple way to run the pipeline.
    # It defines paths for input and output, attempts to load and process data,
    # and uses a sample DataFrame if the initial loading fails.

    sample_file_path = "data/non_existent_payments.csv"  # Example input CSV path
    processed_output_path = "data/processed_payments.parquet" # Example output Parquet path

    # Attempt to load the raw data from CSV
    df_loaded = load_csv_data(sample_file_path)

    final_df_to_process = None

    if df_loaded is not None:
        # If CSV loading is successful, use a copy of the loaded DataFrame
        final_df_to_process = df_loaded.copy()
        print("Successfully loaded data from CSV for preprocessing.")
    else:
        # If CSV loading fails (e.g., file not found), use a predefined sample DataFrame for demonstration.
        print("Failed to load data from CSV. Using sample DataFrame for preprocessing demonstration.")
        sample_data = {
            'transaction_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'transaction_date': [
                '01/15/2024', '01/15/2024', None, '01/17/2024',
                '05/30/2025', '05/30/2025', '05/31/2025',
                'bad_date', '02/29/2023', # Invalid date to test coerce
                '03/01/2024'
            ], # M/D/YYYY format, includes future dates, duplicates, None, bad_date
            'total': [500, 750.25, 2000, 1200.50, 15000, 8000, 7500, 600, 900, 120.75], # Larger totals
            'balance_remaining': [0, 50.50, 100.0, 20.0, 5000, 0, 750.00, 60, 90, 0],
            'payment_type': ['Credit Card', 'Cash', 'Debit Card', None, 'Credit Card', 'Online', 'Check', 'Cash', 'Credit Card', 'Debit Card'],
            'voided': [0, 'f', True, '1', 'FALSE', None, 't', 'F', 0, 1], # Diverse voided values
            'time_created': [
                '2024-01-15 10:00:00', '2024-01-15 10:05:00', '2024-01-16 12:00:00',
                '2024-01-17 13:00:00', '2025-05-30 14:00:00', '2025-05-30 14:30:00',
                '2025-05-31 15:00:00', '2024-01-18 10:00:00', '2023-02-29 11:00:00', # time for bad date
                '2024-03-01 12:00:00'
                ],
            'time_modified': [
                '2024-01-15 10:00:00', '2024-01-15 11:05:00', '2024-01-16 12:00:00',
                '2024-01-17 13:05:00', '2025-05-30 14:05:00', '2025-05-30 14:35:00',
                '2025-05-31 15:10:00', '2024-01-18 10:00:00', '2023-02-29 11:00:00',
                '2024-03-01 12:05:00'
                ],
            'time_voided': [
                None, None, '2024-01-16 12:30:00', None, None, None, '2025-05-31 15:15:00',
                None, None, '2024-03-01 12:10:00'
                ]
        }
        df_sample_for_processing = pd.DataFrame(sample_data)

        # Explicitly convert date/time columns in the sample DataFrame,
        # simulating how load_csv_data would process them.
        # This ensures preprocess_data receives data in the expected format.
        if 'transaction_date' in df_sample_for_processing.columns:
            df_sample_for_processing['transaction_date'] = pd.to_datetime(
                df_sample_for_processing['transaction_date'], format='%m/%d/%Y', errors='coerce'
            )
        other_date_cols_sample = ['time_created', 'time_modified', 'time_voided']
        for col in other_date_cols_sample:
            if col in df_sample_for_processing.columns:
                 df_sample_for_processing[col] = pd.to_datetime(
                     df_sample_for_processing[col], errors='coerce'
                 )
        final_df_to_process = df_sample_for_processing

    # Ensure there is a DataFrame to process
    if final_df_to_process is not None:
        print("\nDataFrame before preprocessing (sample or loaded):")
        print(final_df_to_process.head())
        # final_df_to_process.info() # .info() prints to stdout, can be verbose for automation

        # Preprocess the DataFrame
        df_processed = preprocess_data(final_df_to_process)

        print("\nProcessed DataFrame head:")
        print(df_processed.head())
        # df_processed.info() # .info() prints to stdout

        # Save the processed DataFrame to Parquet
        if not df_processed.empty: # Only save if the processed DataFrame is not empty
            print(f"\nAttempting to save processed data to {processed_output_path}...")
            save_processed_data(df_processed, processed_output_path)
        else:
            print("\nProcessed DataFrame is empty. Skipping save operation.")
    else:
        # This case should ideally not be reached if sample data is always created as a fallback.
        print("No data available for processing.")
