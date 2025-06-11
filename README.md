# Payment Receipt Projection System

## Overview

This project aims to project payment receipts on a daily basis using historical transaction data. It leverages Python for data processing and the Prophet forecasting library by Facebook for time series prediction. The primary goal is to provide a structured pipeline for data loading, preprocessing, model training, and prediction.

Core technologies used:
- Python
- pandas for data manipulation
- Prophet for time series forecasting
- scikit-learn for evaluation metrics
- joblib for model serialization

## Project Structure

The project is organized into the following main directories:

-   `data/`: Contains raw input data (CSV) and processed data (Parquet).
    -   `payments.csv` (example): Raw input CSV file (user needs to provide this).
    -   `processed_payments.parquet`: Processed data output by `load_data.py`.
-   `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and experimentation.
    -   `eda.ipynb`: Example notebook for initial data exploration.
-   `src/`: Contains the Python source code for the project.
    -   `src/pipelines/`: Modules for different stages of the MLOps pipeline.
        -   `load_data.py`: Handles loading raw data, preprocessing, and saving processed data.
        -   `train_model.py`: Trains the Prophet forecasting model on processed data.
        -   `predict.py`: Loads the trained model to make future predictions and evaluate.
    -   `src/utils/`: (Placeholder) Intended for utility functions (currently not used).
-   `tests/`: Contains unit tests for the pipeline scripts.
    -   `test_load_data.py`: Tests for the data loading and preprocessing pipeline.
    -   `test_train_model.py`: Tests for the model training pipeline.
    -   `test_predict.py`: Tests for the prediction pipeline.
-   `models/`: Stores serialized trained machine learning models.
    -   `prophet_model.joblib`: Trained Prophet model saved by `train_model.py`.

**Key Files:**

-   `requirements.txt`: Lists all Python dependencies for the project.

## Setup and Installation

1.  **Create a Virtual Environment (Recommended):**
    It's highly recommended to create and activate a virtual environment to manage project dependencies in isolation.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    Install all required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    This will install pandas, Prophet, scikit-learn, joblib, and their dependencies.

## Data

The system expects payment data in a CSV file.

**Expected CSV Structure:**
The CSV file should contain columns relevant to payment transactions. Key columns used by the current pipeline include:
-   `transaction_id` (or any unique ID)
-   `transaction_date`: Date of the transaction (e.g., "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"). This is critical for time series forecasting.
-   `total`: The total amount of the payment.
-   `voided`: Indicator if the transaction was voided (e.g., True/False, 1/0, 't'/'f').
-   `payment_type`: Type of payment (e.g., 'Credit Card', 'Cash').
-   `time_created`, `time_modified`, `time_voided`: Additional datetime fields that are loaded but might require specific feature engineering if used directly in modeling.

**Data Placement:**
-   Place your raw input CSV file (e.g., `payments.csv`) in the `data/` directory.
-   The `load_data.py` script, if it doesn't find the specified CSV, will use an internal sample DataFrame for demonstration.
-   After processing, the script saves the cleaned data as `data/processed_payments.parquet`.

## Running the Pipeline

The pipeline is executed by running individual Python scripts in sequence. Ensure you are in the root directory of the project when running these commands.

1.  **Data Preprocessing:**
    This step loads the raw CSV data (or uses a sample if not found), preprocesses it (handles missing values, engineers features, converts types), and saves the result to `data/processed_payments.parquet`.
    ```bash
    python src/pipelines/load_data.py
    ```

2.  **Model Training:**
    This step loads the `data/processed_payments.parquet` file, prepares the data for Prophet (aggregates daily totals for non-voided transactions), trains the model, and saves the trained model to `models/prophet_model.joblib`.
    ```bash
    python src/pipelines/train_model.py
    ```

3.  **Prediction:**
    This step loads the trained model from `models/prophet_model.joblib`, generates a forecast for a defined future period (e.g., 30 days), prints the forecast, and performs a simplified in-sample evaluation.
    ```bash
    python src/pipelines/predict.py
    ```

## Running Tests

Unit tests are provided to ensure the functionality of individual pipeline components.

-   **Run all tests:**
    Navigate to the project root directory and run:
    ```bash
    python -m unittest discover tests
    ```
-   **Run a specific test file:**
    For example, to run tests for the `load_data.py` script:
    ```bash
    python -m unittest tests.test_load_data.py
    ```
    Similarly for `tests.test_train_model` and `tests.test_predict`.

## Future Enhancements

This project provides a basic framework. Potential areas for improvement and expansion include:

-   **Advanced Feature Engineering:** Incorporate more sophisticated features (e.g., holidays, external regressors, payment type patterns).
-   **Model Experimentation:** Try different forecasting models (e.g., ARIMA, Exponential Smoothing, other ML models like XGBoost adapted for time series).
-   **Hyperparameter Tuning:** Implement systematic hyperparameter tuning for the Prophet model (or other models).
-   **Robust Evaluation:** Implement proper out-of-sample evaluation using techniques like rolling-origin cross-validation (time series cross-validation).
-   **Configuration Management:** Use configuration files (e.g., YAML, JSON) to manage paths, model parameters, and other settings.
-   **MLOps Integration:** Integrate with MLOps tools for experiment tracking (e.g., MLflow, Weights & Biases), versioning, and deployment.
-   **API for Predictions:** Develop a simple API (e.g., using Flask or FastAPI) to serve predictions on demand.
-   **Error Handling and Logging:** Enhance error handling and implement more structured logging throughout the pipelines.
-   **Parameterization:** Allow command-line arguments for scripts (e.g., input/output paths, forecast periods).
