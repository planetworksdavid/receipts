"""
Handles the training of a Prophet forecasting model.

This module provides functions to:
- Load processed payment data from a Parquet file.
- Prepare the data specifically for Prophet (column renaming, aggregation).
- Train a Prophet model on the prepared time-series data.
- Save the trained model to a file using joblib for later use in predictions.
"""
import pandas as pd
from prophet import Prophet
import joblib
import os

def load_processed_data(file_path: str) -> pd.DataFrame | None:
    """
    Reads a Parquet file containing processed data into a pandas DataFrame.

    :param file_path: The path to the Parquet file.
    :type file_path: str
    :return: A pandas DataFrame with the processed data, or None if an error
             (e.g., FileNotFoundError, issues during reading) occurs.
    :rtype: pd.DataFrame or None
    """
    try:
        df = pd.read_parquet(file_path)
        # Attempt to read the Parquet file
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded processed data from {file_path}")
        return df
    except FileNotFoundError:
        # Handle cases where the file does not exist
        print(f"Error: Processed data file not found at {file_path}")
        return None
    except Exception as e:
        # Handle other potential errors during file reading
        print(f"Error loading processed data from {file_path}: {e}")
        return None

def train_prophet_model(df: pd.DataFrame) -> Prophet | None:
    """
    Prepares data and trains a Prophet time-series forecasting model.

    The function performs the following steps:
    1. Checks for empty or invalid input DataFrame and required columns
       ('transaction_date', 'total', 'is_voided').
    2. Filters out transactions that are marked as voided (`is_voided` is True).
    3. Renames 'transaction_date' to 'ds' and 'total' to 'y' as required by Prophet.
    4. Aggregates the 'y' values by summing them per day ('ds').
    5. Initializes a Prophet model instance.
    6. Fits the model to the prepared daily aggregated data.

    :param df: A pandas DataFrame containing processed transaction data.
               Must include 'transaction_date' (datetime), 'total' (numeric),
               and 'is_voided' (boolean) columns.
    :type df: pd.DataFrame
    :return: A trained `prophet.Prophet` model instance if training is successful.
             Returns None if the input data is unsuitable (e.g., empty, missing required columns,
             no non-voided transactions, or data aggregation results in an empty set)
             or if any other exception occurs during training.
    :rtype: prophet.Prophet or None
    """
    # Validate input DataFrame
    if df is None or df.empty:
        print("Error: Input DataFrame is empty. Cannot train model.")
        return None

    required_columns = ['transaction_date', 'total', 'is_voided']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: DataFrame must contain the columns: {', '.join(required_columns)} for training.")
        return None

    try:
        print("Preparing data for Prophet model...")
        # 1. Filter out voided transactions to focus on actual payments received.
        df_filtered = df[~df['is_voided']].copy()

        if df_filtered.empty:
            print("Error: No non-voided transactions found in the input data. Cannot train model.")
            return None

        # 2. Rename columns to 'ds' (datestamp) and 'y' (target value) as required by Prophet.
        df_prophet = df_filtered.rename(columns={'transaction_date': 'ds', 'total': 'y'})

        # 3. Aggregate transaction data to daily totals. Prophet works best with daily data.
        # Group by 'ds' (which is now the transaction date) and sum 'y' (total amount).
        df_prophet_daily = df_prophet.groupby('ds')['y'].sum().reset_index()

        if df_prophet_daily.empty:
            print("Error: Data aggregation resulted in an empty DataFrame. Ensure there's data to aggregate.")
            return None

        # Basic check for sufficient data points (Prophet needs at least 2 distinct points)
        if len(df_prophet_daily) < 2:
            print(f"Error: Insufficient data points ({len(df_prophet_daily)}) after aggregation for Prophet model training. Need at least 2.")
            return None

        print(f"Prophet input data summary (after daily aggregation):\n{df_prophet_daily.head()}")
        print(f"Number of data points for Prophet: {len(df_prophet_daily)}")

        # 4. Initialize and fit the Prophet model.
        # Default Prophet settings are used here. Further customization (e.g., seasonality, holidays)
        # could be added based on specific requirements and data characteristics.
        model = Prophet()
        print("Training Prophet model...")
        model.fit(df_prophet_daily) # Fit the model using the prepared daily data
        print("Prophet model training complete.")
        return model

    except Exception as e:
        # Catch any other unforeseen errors during the training process
        print(f"An unexpected error occurred during Prophet model training: {e}")
        return None

def save_model(model: Prophet, model_path: str) -> None:
    """
    Saves the trained Prophet model to a specified path using joblib.

    This function creates the target directory if it doesn't exist.

    :param model: The trained model object to be saved (expected to be a Prophet model).
    :type model: prophet.Prophet
    :param model_path: The file path (including filename and .joblib extension)
                       where the model should be saved.
    :type model_path: str
    :return: None
    :rtype: None
    :raises Exception: Prints an error message if any exception occurs during model saving.
    """
    try:
        # Ensure the directory for the model path exists; create it if not.
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir) and model_dir: # Check if model_dir is not empty string
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created directory for model: {model_dir}")

        joblib.dump(model, model_path) # Use joblib to serialize and save the model
        print(f"Model successfully saved to {model_path}")
    except Exception as e:
        # Catch potential errors during directory creation or model saving
        print(f"Error saving model to {model_path}: {e}")

if __name__ == "__main__":
    # This block demonstrates the usage of the functions in this module
    # as part of a simple model training pipeline execution.

    processed_data_path = "data/processed_payments.parquet"  # Input: path to processed data
    model_output_path = "models/prophet_model.joblib"      # Output: path to save the trained model

    print("--- Starting Model Training Pipeline ---")

    # Step 1: Load processed data
    print(f"\nStep 1: Loading processed data from '{processed_data_path}'...")
    df_processed = load_processed_data(processed_data_path)

    if df_processed is not None and not df_processed.empty:
        # Step 2: Train Prophet model using the loaded processed data
        print("\nStep 2: Training Prophet model...")
        trained_model = train_prophet_model(df_processed)

        if trained_model is not None:
            # Step 3: Save the successfully trained model
            print(f"\nStep 3: Saving trained model to '{model_output_path}'...")
            save_model(trained_model, model_output_path)
        else:
            # Handle cases where model training did not succeed
            print("Model training failed. Skipping model saving.")
    else:
        # Handle cases where data loading failed or returned an empty DataFrame
        print("Failed to load processed data or data is empty. Aborting model training pipeline.")

    print("\n--- Model Training Pipeline Finished ---")
