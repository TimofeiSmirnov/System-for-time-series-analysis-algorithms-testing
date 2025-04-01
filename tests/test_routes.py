import os
import sys
import io
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app, create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_generate_series_with_valid_parameters(client):
    data = {
        'n': 500,
        'k': 10,
        'avg_pattern_length': 50,
        'avg_amplitude': 0.5,
        'default_variance': 0.1,
        'variance_pattern_length': 30,
        'variance_amplitude': 0.2,
        'algorithm_for_gen': 'stomp',
        'm_for_gen': 100
    }
    response = client.post('/matrix_profile', data=data)
    assert response.status_code == 200
    assert b"Matrix Profile Calculation for Time Series" in response.data


def test_generate_series_with_invalid_parameters(client):
    data = {
        'n': 50,
        'k': 10,
        'avg_pattern_length': 50,
        'avg_amplitude': 0.5,
        'default_variance': 0.1,
        'variance_pattern_length': 30,
        'variance_amplitude': 0.2,
        'algorithm_for_gen': 'stomp',
        'm_for_gen': 100
    }
    response = client.post('/matrix_profile', data=data)
    assert response.status_code == 200
    assert b"The time series is too short" in response.data


def test_upload_time_series_file(client):
    data = {
        'file': (io.BytesIO(b'col1,col2\n1,2\n3,4\n5,6'), 'test_series.csv'),
        'algorithm_for_load': 'stomp',
        'm_for_load': 100
    }
    response = client.post('/matrix_profile', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"Matrix Profile Calculation for Time Series" in response.data


def test_upload_time_series_file_with_nan(client):
    data = {
        'file': (io.BytesIO(b'col1,col2\n1,2\n3,4\n5,NaN'), 'test_series_with_nan.csv'),
        'algorithm_for_load': 'stomp',
        'm_for_load': 100
    }
    response = client.post('/matrix_profile', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"The uploaded series contains missing or NaN values" in response.data


def test_load_and_generate_matrix_profile(client):
    df = pd.DataFrame({'timestamp': [1, 2, 3], 'value': [10, 20, 30]})
    file_stream = io.BytesIO()
    df.to_csv(file_stream, index=False)
    file_stream.seek(0)

    data = {
        'file': (file_stream, 'test_series.csv'),
        'algorithm_for_load': 'stomp',
        'm_for_load': 2
    }

    response = client.post('/matrix_profile', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"Matrix Profile Calculation for Time Series" in response.data
