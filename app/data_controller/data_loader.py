import numpy as np
import random
import math


def generate_bell(length, amplitude, default_variance):
    """Генерирует восходящий паттерн"""
    bell = np.random.normal(0, default_variance, length) + amplitude * np.arange(length)/length
    return bell


def generate_funnel(length, amplitude, default_variance):
    """Генерирует нисходящий паттерн"""
    funnel = np.random.normal(0, default_variance, length) + amplitude * np.arange(length)[::-1]/length
    return funnel


def generate_cylinder(length, amplitude, default_variance):
    """Генерирует горизонтальный паттерн"""
    cylinder = np.random.normal(0, default_variance, length) + amplitude
    return cylinder


std_generators = [generate_bell, generate_funnel, generate_cylinder]


def generate_ts_data(length=100, avg_pattern_length=5, avg_amplitude=1,
                          default_variance=1, variance_pattern_length=10, variance_amplitude=2,
                          generators=std_generators, include_negatives=False):
    """Генерирует временной ряд"""
    data = np.random.normal(0, default_variance, length)
    current_start = random.randint(0, avg_pattern_length)
    current_length = max(1, math.ceil(random.gauss(avg_pattern_length, variance_pattern_length)))

    while current_start + current_length < length:
        generator = random.choice(generators)
        current_amplitude = random.gauss(avg_amplitude, variance_amplitude)

        while current_length <= 0:
            current_length = -(current_length - 1)


        pattern = generator(current_length, current_amplitude, default_variance)

        data[current_start: current_start + current_length] = pattern

        current_start = current_start + current_length + random.randint(0, avg_pattern_length)
        current_length = max(1, math.ceil(random.gauss(avg_pattern_length, variance_pattern_length)))

    return np.array(data)


def generate_timestamps(n):
    """Создает последовательность временных меток"""
    return np.arange(n).astype(np.float64)


def generate_time_series(n, k):
    """Сейчас генерирует случайный ряд, но скоро можно будет его загружать из базы"""
    values = []

    for i in range(k):
        values.append(generate_ts_data(length=n, avg_pattern_length=20, default_variance=2))

    return generate_timestamps(n), values