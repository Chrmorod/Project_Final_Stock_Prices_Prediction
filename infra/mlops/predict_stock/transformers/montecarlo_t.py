try:
    from mage_ai.data_preparation.decorators import transformer
except ImportError:

    def transformer(func):
        return func


import mlflow
import cloudpickle
import os
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import warnings


@transformer
def transform(data, *args, **kwargs):

    warnings.simplefilter(action="ignore", category=FutureWarning)

    data.index = pd.to_datetime(data.index)

    n_days = int(kwargs.get("n_days", 30))
    n_simulations = int(kwargs.get("n_simulations", 1000))

    prices = data["Close"]
    last_price = float(prices.iloc[-1])

    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    dt = 1 / 252

    def stock_monte_carlo(start_price, days, mu, sigma):
        price = np.zeros(days)
        price[0] = start_price
        for t in range(1, days):
            shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
            drift = mu * dt
            price[t] = price[t - 1] + (price[t - 1] * (drift + shock))
        return price

    simulations = np.zeros((n_days, n_simulations))
    for i in range(n_simulations):
        simulations[:, i] = stock_monte_carlo(last_price, n_days, mu, sigma)

    last_date = prices.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=n_days
    )
    simulations_df = pd.DataFrame(simulations, index=future_dates)

    last_simulated_price = simulations_df.iloc[-1]
    average_expected_price = last_simulated_price.mean()
    conf_interval = np.percentile(last_simulated_price, [5, 95])

    montecarlo_model_data = {
        "simulations_df": simulations_df,
        "parameters": {
            "n_days": n_days,
            "n_simulations": n_simulations,
            "mu": mu,
            "sigma": sigma,
            "last_price": last_price,
        },
    }

    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment(
            f"montecarlo_{n_days}d_{n_simulations}s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        with mlflow.start_run(run_name=f"montecarlo_{n_days}d_{n_simulations}s"):
            mlflow.log_param("n_days", n_days)
            mlflow.log_param("n_simulations", n_simulations)
            mlflow.log_param("expected_price", average_expected_price)
            mlflow.log_param("mu", mu)
            mlflow.log_param("sigma", sigma)

            mlflow.log_metric("average_expected_price", average_expected_price)
            mlflow.log_metric("conf_interval_low", conf_interval[0])
            mlflow.log_metric("conf_interval_high", conf_interval[1])

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                temp_file_path = tmp.name
                cloudpickle.dump(montecarlo_model_data, tmp)

            artifact_path = "montecarlo_model"
            mlflow.log_artifact(temp_file_path, artifact_path=artifact_path)
            os.remove(temp_file_path)

            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/{artifact_path}"

            registered_model_name = "Monte_Carlo_Stock_Predictor"

            mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            print(f"IC 90%: ${conf_interval[0]:.2f} - ${conf_interval[1]:.2f}")

    except Exception as e:
        print(f"MLflow error: {e}")

    return {
        "last_expected_price_montecarlo": average_expected_price,
        "average_expected_price_montecarlo": average_expected_price,
    }
