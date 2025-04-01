import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app, create_app
from app.utils.mp_calculator import generate_mp, arc_curve_calculator

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_generate_mp():
    """Тестируем генерацию матричного профиля"""
    time_series = [np.random.rand(100).tolist()]
    window_length = 10

    algorithms = ["stomp", "scrimp++", "stump", "mstump"]

    for algo in algorithms:
        result, execution_time = generate_mp(algo, time_series, window_length)

        assert len(result) > 0
        assert execution_time > 0

    with pytest.raises(ValueError, match="Unknown algo"):
        generate_mp("unknown_algo", time_series, window_length)


def test_arc_curve_calculator():
    """Тестируем генерацию CAC"""
    time_series = np.random.rand(100)
    m = 10
    n_regimes = 3

    result = arc_curve_calculator(time_series, m, n_regimes)

    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)
    assert len(result[0]) == len(time_series) - m + 1
