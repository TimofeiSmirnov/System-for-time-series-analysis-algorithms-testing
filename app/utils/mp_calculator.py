import time
import stumpy
from matrixprofile import matrixProfile
from dask.distributed import Client


def generate_mp(algorithm_type, time_series, m):
    """Вычисляет матричный профиль в зависимости от запроса пользователя и замеряет время выполнения"""
    start_time = time.time()

    if algorithm_type == "stomp":
        result = [matrixProfile.stomp(time_series[0], m)[0]]
    elif algorithm_type == "scrimp++":
        result = [matrixProfile.scrimp_plus_plus(time_series[0], m)[0]]
    elif algorithm_type == "stump":
        result = stumpy.stump(time_series[0], m=m)
    elif algorithm_type == "mstump":
        result = stumpy.mstump(time_series, m=m)[0]
    else:
        raise ValueError(f"Неизвестный алгоритм: {algorithm_type}")

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
