# test_export_data.py

import pytest
from unittest.mock import patch
import sys
import os
import sys



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infra.mlops.predict_stock.data_exporters import export_data_t as export_module
export_data = export_module.export_data

@pytest.fixture
def mock_data():
    return {
        'last_expected_price_montecarlo': 100.0,
        'average_expected_price_montecarlo': 95.0
    }

@pytest.fixture
def mock_data_2():
    return {
        'last_expected_price_arima': 110.0,
        'average_expected_price_arima': 105.0
    }

@pytest.fixture
def mock_data_3():
    return {
        'last_expected_price_lstm': 120.0,
        'average_expected_price_lstm': 115.0
    }

def test_export_data(mock_data, mock_data_2, mock_data_3):
    with patch("infra.mlops.predict_stock.data_exporters.export_data_t.mlflow") as mock_mlflow:
        result = export_data(mock_data, mock_data_2, mock_data_3)

        expected_hybrid_price = (100.0 + 110.0 + 120.0) / 3
        expected_avg_price = (95.0 + 105.0 + 115.0) / 3

        assert result['hybrid_expected_price'] == expected_hybrid_price
        assert result['hybrid_average_expected_price'] == expected_avg_price
