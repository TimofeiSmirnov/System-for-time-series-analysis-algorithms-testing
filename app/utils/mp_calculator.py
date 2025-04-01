import time
import stumpy
from matrixprofile import matrixProfile
import numpy as np


def generate_mp(
    algorithm_type: str,
    time_series: list[list[float]],
    window_length: int
) -> tuple[list[np.ndarray], float]:
    """
    Вычисляет матричный профиль в зависимости от запроса пользователя и замеряет время выполнения
    :param algorithm_type: (str) тип алгоритма
    :param time_series: (list[list[float]]) временной ряд
    :param window_length: (int) длина окна для вычисления матричного профиля.
    :return:
        - result (list[numpy.ndarray]): список массивов с матричным профилем, рассчитанным для каждого ряда.
        - execution_time (float): Время выполнения алгоритма в секундах.
    """
    start_time = time.time()
    if algorithm_type == "stomp":
        result = [matrixProfile.stomp(time_series[0], window_length)[0]]
    elif algorithm_type == "scrimp++":
        result = [matrixProfile.scrimp_plus_plus(np.array(time_series[0]), window_length)[0]]
    elif algorithm_type == "stump":
        result = stumpy.stump(np.array(time_series[0]), m=window_length)
    elif algorithm_type == "mstump":
        result = stumpy.mstump(np.array(time_series), m=window_length)[0]
    else:
        raise ValueError(f"Unknown algo: {algorithm_type}")

    end_time = time.time()

    if algorithm_type == "stump" or algorithm_type == "stumped":
        result = [[result_chunk[0] for result_chunk in result]]

    execution_time = end_time - start_time

    return result, execution_time


def arc_curve_calculator(time_series, m, n_regimes):
    """Высчитываем арочную кривую"""
    matrix_profile = stumpy.stump(time_series, m=m)
    cac, regime_locations = stumpy.fluss(matrix_profile[:, 1], L=m, n_regimes=n_regimes, excl_factor=1)
    return [cac]
