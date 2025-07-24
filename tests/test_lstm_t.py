# tests/test_lstm.py

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infra.mlops.predict_stock.transformers.lstm_t import transform

@pytest.fixture
def fake_price_data():
    dates = pd.bdate_range(start="2023-01-01", periods=200)
    close_prices = np.linspace(100, 200, 200)
    return pd.DataFrame({"Close": close_prices}, index=dates)

@patch("infra.mlops.predict_stock.transformers.lstm_t.mlflow.register_model")
@patch("infra.mlops.predict_stock.transformers.lstm_t.mlflow.keras.log_model")
@patch("infra.mlops.predict_stock.transformers.lstm_t.mlflow.start_run")
@patch("infra.mlops.predict_stock.transformers.lstm_t.mlflow.set_experiment")
@patch("infra.mlops.predict_stock.transformers.lstm_t.mlflow.set_tracking_uri")
@patch("infra.mlops.predict_stock.transformers.lstm_t.Sequential.fit")
@patch("infra.mlops.predict_stock.transformers.lstm_t.Sequential.predict")
def test_transform_lstm(mock_predict, mock_fit, mock_set_uri, mock_set_exp, mock_start_run, mock_log_model, mock_register_model, fake_price_data):
    fake_preds = np.linspace(0.5, 1.0, 30).reshape(-1, 1)
    mock_predict.side_effect = [np.array([[val]]) for val in fake_preds]

    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"
    mock_start_run.return_value.__enter__.return_value = mock_run

    result = transform(fake_price_data)

    assert isinstance(result["last_expected_price_lstm"], float)
    assert isinstance(result["average_expected_price_lstm"], float)

    assert 100 < result["last_expected_price_lstm"] < 250
    assert 100 < result["average_expected_price_lstm"] < 250

    mock_set_uri.assert_called_once()
    mock_log_model.assert_called_once()
    mock_register_model.assert_called_once()
