# test_extract_data.py

import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from infra.mlops.predict_stock.data_loaders.data_extract_t import (  # noqa: E402
    extract_data,  # noqa: E402
)  # noqa: E402


@pytest.fixture
def mock_yf_data():
    return pd.DataFrame(
        {"Close": [100, 101, 102], "Volume": [1000, 1100, 1050]},
        index=pd.date_range(start="2021-01-01", periods=3),
    )


@patch("infra.mlops.predict_stock.data_loaders.data_extract_t.yf")
@patch("infra.mlops.predict_stock.data_loaders.data_extract_t.mlflow")
def test_extract_data(mock_mlflow, mock_yf, mock_yf_data):
    mock_yf.download.return_value = mock_yf_data

    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.set_experiment = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_mlflow.log_param = MagicMock()
    mock_mlflow.log_metric = MagicMock()

    result = extract_data(stock="AAPL", year_back=1)

    assert isinstance(result, pd.DataFrame)
    assert "Close" in result.columns
    assert result.shape[0] == 3
