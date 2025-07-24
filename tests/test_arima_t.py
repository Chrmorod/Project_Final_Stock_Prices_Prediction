# test_arima.py

import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from infra.mlops.predict_stock.transformers.arima_t import transform

@pytest.fixture
def fake_price_data():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    close_prices = np.linspace(100, 150, 100)
    return pd.DataFrame({"Close": close_prices}, index=dates)

@patch("infra.mlops.predict_stock.transformers.arima_t.tempfile.NamedTemporaryFile")
@patch("infra.mlops.predict_stock.transformers.arima_t.cloudpickle.dump")
@patch("infra.mlops.predict_stock.transformers.arima_t.mlflow")
def test_transform_arima(mock_mlflow, mock_pickle, mock_tempfile, fake_price_data):
    mock_tmp = MagicMock()
    mock_tmp.name = "fake_path.pkl"
    mock_tempfile.return_value.__enter__.return_value = mock_tmp

    result = transform(fake_price_data)
    expected_last = 165.15081265666382
    expected_avg = 157.82801896627748
    
    assert isinstance(result["last_expected_price_arima"], float)
    assert isinstance(result["average_expected_price_arima"], float)
    assert abs(result["last_expected_price_arima"] - expected_last) < 1e-6
    assert abs(result["average_expected_price_arima"] - expected_avg) < 1e-6