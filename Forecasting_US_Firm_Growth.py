import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import warnings

warnings.filterwarnings("ignore")

# Load the data with the correct file path
file_path = (
    "/Users/rudolphstockenstrom/Documents/Data projects/P3 - Forecasting/us_data.csv"
)

try:
    df = pd.read_csv(file_path)
    print("File successfully loaded!")
except FileNotFoundError:
    print(f"Error: Could not find the file at {file_path}")
    exit()

# Convert year to datetime
df["year"] = pd.to_datetime(df["year"], format="%Y")
df.set_index("year", inplace=True)


# Function to check stationarity
def check_stationarity(series, title):
    result = adfuller(series.dropna())
    print(f"Stationarity Test for {title}")
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value}")
    print("\n")


# Function to create ARIMA model and forecast
def create_arima_forecast(data, column, order=(1, 1, 1)):
    # Split data into train and test
    train_size = int(len(data) * 0.8)
    train = data[:train_size]
    test = data[train_size:]

    # Fit ARIMA model
    model = ARIMA(train[column], order=order)
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.forecast(steps=len(test))

    # Calculate error metrics
    rmse = sqrt(mean_squared_error(test[column], predictions))
    mae = mean_absolute_error(test[column], predictions)

    # Future predictions (5 years ahead)
    future_dates = pd.date_range(start=data.index[-1], periods=6, freq="Y")[1:]
    future_predictions = model_fit.forecast(steps=5)

    return (
        train,
        test,
        predictions,
        future_predictions,
        future_dates,
        rmse,
        mae,
        model_fit,
    )


# Initial data analysis
print("\nDataset Overview:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Select key metrics for analysis
key_metrics = [
    "firms",
    "job_creation_rate",
    "job_destruction_rate",
    "net_job_creation_rate",
]

# Create subplots for key metrics trends
plt.figure(figsize=(15, 10))
for i, metric in enumerate(key_metrics, 1):
    plt.subplot(2, 2, i)
    plt.plot(df.index, df[metric])
    plt.title(f'{metric.replace("_", " ").title()} Over Time')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze and forecast business growth (using number of firms)
print("\nAnalyzing Business Growth (Number of Firms)")
check_stationarity(df["firms"], "Number of Firms")

train, test, predictions, future_predictions, future_dates, rmse, mae, model_firms = (
    create_arima_forecast(df, "firms")
)

# Plot results for firms
plt.figure(figsize=(12, 6))
plt.plot(train.index, train["firms"], label="Training Data")
plt.plot(test.index, test["firms"], label="Actual Values")
plt.plot(test.index, predictions, label="Predictions")
plt.plot(future_dates, future_predictions, label="Future Forecast", linestyle="--")
plt.title("Business Growth Forecast (Number of Firms)")
plt.xlabel("Year")
plt.ylabel("Number of Firms")
plt.legend()
plt.show()

# Analyze and forecast job creation
print("\nAnalyzing Job Creation Rate")
check_stationarity(df["job_creation_rate"], "Job Creation Rate")

(
    train_jc,
    test_jc,
    pred_jc,
    future_pred_jc,
    future_dates_jc,
    rmse_jc,
    mae_jc,
    model_jc,
) = create_arima_forecast(df, "job_creation_rate")

# Plot results for job creation rate
plt.figure(figsize=(12, 6))
plt.plot(train_jc.index, train_jc["job_creation_rate"], label="Training Data")
plt.plot(test_jc.index, test_jc["job_creation_rate"], label="Actual Values")
plt.plot(test_jc.index, pred_jc, label="Predictions")
plt.plot(future_dates_jc, future_pred_jc, label="Future Forecast", linestyle="--")
plt.title("Job Creation Rate Forecast")
plt.xlabel("Year")
plt.ylabel("Job Creation Rate")
plt.legend()
plt.show()

# Print forecast results
print("\nForecast Results for Number of Firms:")
future_firms_df = pd.DataFrame(
    {"Year": future_dates, "Predicted_Firms": future_predictions}
)
print(future_firms_df)

print("\nForecast Results for Job Creation Rate:")
future_jc_df = pd.DataFrame(
    {"Year": future_dates_jc, "Predicted_Job_Creation_Rate": future_pred_jc}
)
print(future_jc_df)

# Print model performance metrics
print("\nModel Performance Metrics:")
print(f"Number of Firms - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
print(f"Job Creation Rate - RMSE: {rmse_jc:.4f}, MAE: {mae_jc:.4f}")

# Business dynamics analysis
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["estabs_entry_rate"], label="Establishment Entry Rate")
plt.plot(df.index, df["estabs_exit_rate"], label="Establishment Exit Rate")
plt.title("Business Dynamics: Entry vs Exit Rates")
plt.xlabel("Year")
plt.ylabel("Rate")
plt.legend()
plt.show()

# Correlation analysis of key metrics
correlation_vars = [
    "firms",
    "job_creation_rate",
    "job_destruction_rate",
    "estabs_entry_rate",
    "estabs_exit_rate",
    "net_job_creation_rate",
]
correlation_matrix = df[correlation_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Key Business Indicators")
plt.tight_layout()
plt.show()

# Additional analysis: Net job creation trends
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["net_job_creation_rate"])
plt.axhline(y=0, color="r", linestyle="--")
plt.title("Net Job Creation Rate Over Time")
plt.xlabel("Year")
plt.ylabel("Net Job Creation Rate")
plt.show()
