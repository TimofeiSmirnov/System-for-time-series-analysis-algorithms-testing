import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from app.data_controller.data_loader import TimeSeriesGenerator


def test_generate_timestamps():
    """Проверяем, что временные метки генерируются корректно"""
    generator = TimeSeriesGenerator(n=100, k=3)
    timestamps = generator.generate_timestamps()

    assert len(timestamps) == 100
    assert np.all(timestamps == np.arange(100))


def test_generate_ts_data():
    """Проверяем, что метод generate_ts_data создаёт временной ряд нужного размера и формата"""
    generator = TimeSeriesGenerator(n=100, k=3)
    ts_data = generator.generate_ts_data()

    assert len(ts_data) == 100
    assert isinstance(ts_data, np.ndarray)


def test_generate_time_series():
    """Проверяем, что метод generate_time_series создаёт необходимо число временных рядов, а также длину этих рядов"""
    generator = TimeSeriesGenerator(n=100, k=3)
    timestamps, values = generator.generate_time_series()

    assert len(timestamps) == 100
    assert len(values) == 3

    for series in values:
        assert len(series) == 100
        assert isinstance(series, np.ndarray)


def test_pattern_generators():
    """Проверяем, что генераторы паттернов создают массивы нужного размера"""
    generator = TimeSeriesGenerator(n=100, k=3)

    for pattern_generator in [generator.generate_bell, generator.generate_funnel, generator.generate_cylinder]:
        pattern = pattern_generator(50, 1, 0.1)
        assert len(pattern) == 50
        assert isinstance(pattern, np.ndarray)


def test_variability():
    """Проверяем, что создаваемые ряды не одинаковы при разных запусках"""
    generator = TimeSeriesGenerator(n=100, k=3)
    ts1 = generator.generate_ts_data()
    ts2 = generator.generate_ts_data()

    assert not np.array_equal(ts1, ts2)