try:
    from mage_ai.data_preparation.decorators import transformer
except ImportError:

    def transformer(func):
        return func


import mlflow
import tempfile  # noqa: F401
import cloudpickle  # noqa: F401
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import warnings


@transformer
def transform(data):
    warnings.simplefilter(action="ignore", category=FutureWarning)

    data.index = pd.to_datetime(data.index)

    prices = data["Close"]
    n_days = 30

    model = ARIMA(prices, order=(5, 1, 0))
    model_fit = model.fit()

    future_preds = model_fit.forecast(steps=n_days)
    last_date = prices.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=n_days
    )

    future_forecast = pd.DataFrame({"Date": future_dates, "Forecast": future_preds})
    future_forecast.set_index("Date", inplace=True)

    expected_price = future_preds.iloc[-1]
    average_expected_price = future_preds.mean()
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment(
            f"ARIMA_{n_days}d_{last_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        with mlflow.start_run(run_name=f"Run_{last_date.strftime('%Y-%m-%d')}"):
            mlflow.log_param("model", "ARIMA(5,1,0)")
            mlflow.log_param("n_days", n_days)
            mlflow.log_param("last_date", str(last_date))

            mlflow.log_metric("expected_price", expected_price)
            mlflow.log_metric("average_expected_price", average_expected_price)

            mlflow.statsmodels.log_model(
                statsmodels_model=model_fit,
                artifact_path="arima_model",
                registered_model_name="ARIMA_Stock_Predictor",
            )

        if mlflow.active_run():
            mlflow.end_run()
    except Exception as e:
        print(f"MLflow error: {e}")
    return {
        "last_expected_price_arima": expected_price,
        "average_expected_price_arima": average_expected_price,
    }
