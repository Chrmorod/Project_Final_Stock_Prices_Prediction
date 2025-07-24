# test_montecarlo_t.py

import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infra.mlops.predict_stock.transformers.montecarlo_t import transform

@pytest.fixture
def fake_price_data():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    close_prices = np.linspace(100, 150, 100)
    return pd.DataFrame({"Close": close_prices}, index=dates)

import pytest
from unittest.mock import patch, MagicMock
from infra.mlops.predict_stock.transformers.montecarlo_t import transform

@pytest.mark.usefixtures("fake_price_data")
@patch("infra.mlops.predict_stock.transformers.montecarlo_t.os.remove")
@patch("infra.mlops.predict_stock.transformers.montecarlo_t.tempfile.NamedTemporaryFile")
@patch("infra.mlops.predict_stock.transformers.montecarlo_t.cloudpickle.dump")
@patch("infra.mlops.predict_stock.transformers.montecarlo_t.mlflow")
def test_transform_montecarlo(mock_mlflow, mock_pickle, mock_tempfile, mock_remove, fake_price_data):
    mock_tmp = MagicMock()
    mock_tmp.name = "fake_path.pkl"
    mock_tempfile.return_value.__enter__.return_value = mock_tmp

    result = transform(fake_price_data, n_days=10, n_simulations=100)

    assert isinstance(result["last_expected_price_montecarlo"], float)
    assert isinstance(result["average_expected_price_montecarlo"], float)

    mock_tmp.name = "fake_path.pkl"
