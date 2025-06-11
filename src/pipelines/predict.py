"""
Handles loading a trained Prophet model, making future predictions, and evaluating the model.

This module provides functions to:
- Load a serialized Prophet model from a file (expects a .joblib file).
- Create a future DataFrame for generating predictions.
- Generate forecasts using the loaded model.
- Evaluate model performance using MAE and RMSE metrics.
- A helper function to load and prepare actual (historical) data for in-sample evaluation.
"""
import pandas as pd
from prophet import Prophet # Expected type for the model object
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os # Used in the __main__ block for path definitions

# The load_and_prepare_actuals_for_evaluation function below handles its own data loading.

def load_model(model_path: str) -> Prophet | None:
    """
    Loads a saved Prophet model from a file using joblib.

    :param model_path: The path to the saved model file (e.g., 'model.joblib').
    :type model_path: str
    :return: The loaded Prophet model object, or None if an error occurs
             (e.g., file not found, unpickling error).
    :rtype: prophet.Prophet or None
    """
    try:
        model = joblib.load(model_path)
        print(f"Model successfully loaded from {model_path}")
        # Attempt to load the model from the specified path
        model = joblib.load(model_path)
        print(f"Model successfully loaded from {model_path}")
        return model
    except FileNotFoundError:
        # Handle cases where the model file is not found
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        # Handle other potential errors during model loading (e.g., unpickling issues)
        print(f"Error loading model from {model_path}: {e}")
        return None

def make_future_dataframe(model: Prophet, periods: int, freq: str = 'D') -> pd.DataFrame | None:
    """
    Creates a future DataFrame for generating forecasts with a Prophet model.

    This function calls the Prophet model's internal `make_future_dataframe` method.

    :param model: The trained Prophet model instance.
    :type model: prophet.Prophet
    :param periods: The number of future periods to generate.
    :type periods: int
    :param freq: The frequency of the future periods (e.g., 'D' for daily,
                 'W' for weekly, 'MS' for month start). Defaults to 'D'.
    :type freq: str, optional
    :return: A pandas DataFrame with 'ds' column extending into the future,
             or None if an error occurs.
    :rtype: pd.DataFrame or None
    """
    try:
        # Use the model's method to create the future DataFrame
        future_df = model.make_future_dataframe(periods=periods, freq=freq)
        print(f"Future DataFrame created for {periods} '{freq}' periods.")
        return future_df
    except Exception as e:
        # Catch errors that might occur if the model object is invalid or parameters are wrong
        print(f"Error creating future DataFrame: {e}")
        return None

def predict_forecast(model: Prophet, future_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Generates predictions (forecast) using the trained Prophet model.

    This function calls the Prophet model's `predict` method on a future DataFrame.

    :param model: The trained Prophet model instance.
    :type model: prophet.Prophet
    :param future_df: A pandas DataFrame with a 'ds' column representing dates for
                      which predictions are desired. Typically created by
                      `make_future_dataframe`.
    :type future_df: pd.DataFrame
    :return: A pandas DataFrame containing the forecast. This typically includes columns like
             'ds', 'yhat' (predicted value), 'yhat_lower', 'yhat_upper' (uncertainty intervals),
             and trend/seasonality components. Returns None if `future_df` is None or an
             error occurs during prediction.
    :rtype: pd.DataFrame or None
    """
    # Ensure future_df is provided
    if future_df is None:
        print("Error: Future DataFrame is None. Cannot make predictions.")
        return None
    try:
        # Generate forecast using the model's predict method
        forecast = model.predict(future_df)
        print("Forecast generated successfully.")
        return forecast
    except Exception as e:
        # Catch errors during the prediction process
        print(f"Error during forecast prediction: {e}")
        return None

def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float] | None:
    """
    Calculates Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

    If `y_true` and `y_pred` have different lengths but compatible indices,
    the function attempts to align them using their common index before calculation.
    If inputs are empty, None, or cannot be aligned, it returns None.

    :param y_true: A pandas Series of true target values.
    :type y_true: pd.Series
    :param y_pred: A pandas Series of predicted target values.
    :type y_pred: pd.Series
    :return: A dictionary with 'mae' and 'rmse' keys and their float values if successful.
             Returns None if evaluation cannot be performed (e.g., empty inputs,
             mismatched lengths that cannot be aligned, or other calculation errors).
    :rtype: dict[str, float] or None
    """
    # Check for None or empty inputs
    if y_true is None or y_pred is None or y_true.empty or y_pred.empty:
        print("Error: y_true or y_pred is empty or None. Cannot evaluate.")
        return None

    # Attempt to align series if lengths are different, using their index
    if len(y_true) != len(y_pred):
        print(f"Warning: y_true (len {len(y_true)}) and y_pred (len {len(y_pred)}) have different lengths. Attempting to align on common index.")
        common_index = y_true.index.intersection(y_pred.index)
        if len(common_index) == 0:
            print("Error: No common index found between y_true and y_pred for alignment.")
            return None
        y_true_aligned = y_true.loc[common_index]
        y_pred_aligned = y_pred.loc[common_index]

        if y_true_aligned.empty or y_pred_aligned.empty:
             print("Error: Alignment resulted in one or both series being empty.")
             return None
        print(f"Aligned series to common length: {len(y_true_aligned)}")
        y_true, y_pred = y_true_aligned, y_pred_aligned # Use aligned series

    try:
        # Calculate MAE and RMSE
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        return {'mae': mae, 'rmse': rmse}
    except Exception as e:
        # Catch any errors during metric calculation
        print(f"Error during model evaluation metrics calculation: {e}")
        return None

def load_and_prepare_actuals_for_evaluation(processed_data_path: str) -> pd.DataFrame | None:
    """
    Loads and prepares historical processed data for model evaluation.

    This function is typically used to create an "actuals" DataFrame that can be
    compared against a model's historical predictions (in-sample evaluation).
    It reads a Parquet file, filters out voided transactions, renames columns
    to 'ds' (transaction_date) and 'y' (sum of 'total' per day), and ensures
    'ds' is in datetime format.

    :param processed_data_path: Path to the Parquet file containing processed data.
    :type processed_data_path: str
    :return: A pandas DataFrame with 'ds' and 'y' columns ready for evaluation,
             or None if data loading or preparation fails.
    :rtype: pd.DataFrame or None
    """
    try:
        df_processed = pd.read_parquet(processed_data_path)
        if df_processed is None or df_processed.empty:
            print(f"Error: No data loaded from {processed_data_path} for actuals evaluation.")
            return None

        # Filter out voided transactions and select relevant columns
        actuals = df_processed[~df_processed['is_voided']].copy()
        if actuals.empty:
            print("Error: No non-voided transactions found in processed data for actuals.")
            return None

        actuals_prophet = actuals.rename(columns={'transaction_date': 'ds', 'total': 'y'})

        # Aggregate to daily sums for 'y'
        actuals_daily = actuals_prophet.groupby('ds')['y'].sum().reset_index()

        if actuals_daily.empty:
            print("Error: Data aggregation for actuals resulted in an empty DataFrame.")
            return None

        # Ensure 'ds' column is datetime
        actuals_daily['ds'] = pd.to_datetime(actuals_daily['ds'])
        return actuals_daily

    except FileNotFoundError:
        print(f"Error: Actuals data file not found at {processed_data_path}")
        return None
    except Exception as e:
        print(f"Error loading/preparing actuals for evaluation from {processed_data_path}: {e}")
        return None


if __name__ == "__main__":
    # This main block demonstrates a prediction and in-sample evaluation pipeline.

    model_path = "models/prophet_model.joblib" # Path to the saved, trained model
    processed_data_path_for_actuals = "data/processed_payments.parquet" # Path to processed data for historical actuals
    future_periods_to_forecast = 30 # Number of days into the future to forecast

    print("--- Starting Prediction Pipeline ---")

    # Step 1: Load the trained Prophet model
    print(f"\nStep 1: Loading model from '{model_path}'...")
    model = load_model(model_path)

    if model:
        # Step 2: Create a DataFrame for future dates
        print(f"\nStep 2: Creating future DataFrame for {future_periods_to_forecast} days...")
        future_df = make_future_dataframe(model, periods=future_periods_to_forecast, freq='D')

        # Step 3: Generate predictions for the future dates
        if future_df is not None:
            print("\nStep 3: Making predictions...")
            forecast_df = predict_forecast(model, future_df)

            if forecast_df is not None:
                # Display the tail of the forecast, showing the most recent future predictions
                print("\nForecast (tail with predictions):")
                print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                # Step 4: Perform in-sample evaluation (evaluating model on data it was trained on)
                # This is for demonstration; true evaluation requires a separate test set.
                print("\nStep 4: Evaluating model (on in-sample/training data for demonstration)...")
                print(f"Loading actuals for evaluation from '{processed_data_path_for_actuals}'...")
                actuals_df = load_and_prepare_actuals_for_evaluation(processed_data_path_for_actuals)

                if actuals_df is not None and not actuals_df.empty:
                    # Generate predictions for the historical period covered by actuals_df
                    # The model.predict() method can take historical dates too.
                    historical_forecast_df = model.predict(actuals_df[['ds']])

                    if historical_forecast_df is not None and not historical_forecast_df.empty:
                        # Merge actual values with predicted values on the 'ds' (date) column
                        eval_df = pd.merge(actuals_df, historical_forecast_df[['ds', 'yhat']], on='ds', how='inner')

                        if not eval_df.empty:
                            print(f"Evaluation Data (merged actuals and predictions for training period):\n{eval_df.head()}")
                            print(f"Number of data points for in-sample evaluation: {len(eval_df)}")
                            evaluate_model(eval_df['y'], eval_df['yhat'])
                        else:
                            print("Could not merge actuals and historical forecast for evaluation. " \
                                  "Check 'ds' columns and data integrity.")
                    else:
                        print("Failed to generate historical forecast for evaluation.")
                else:
                    print("Failed to load or prepare actuals for evaluation.")
            else:
                print("Forecast generation failed.")
        else:
            print("Future DataFrame creation failed.")
    else:
        # This occurs if the model could not be loaded
        print("Failed to load model. Aborting prediction pipeline.")

    print("\n--- Prediction Pipeline Finished ---")
