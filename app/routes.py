import io
import pandas as pd
from flask import render_template, request, current_app
from app.algorithms.apply_algorithms import ApplyAnomalyDetectionAlgorithms
from app.data_controller.data_controller import DataController
from app.utils.mp_calculator import generate_mp, arc_curve_calculator
from app.utils.visualization import visualize_time_series_with_matrix_profile, visualize_time_series_ad, \
    visualize_time_series_with_fluss, visualize_time_series_ad_multidim


def init_routes(app):
    algorithm_applier = ApplyAnomalyDetectionAlgorithms()
    data_controller = DataController()

    @app.route("/", methods=["GET", "POST"])
    def index():
        """Главная страница с возможностью выбора ряда и генерации матричного профиля по нему"""
        plot_html = None
        available_series = data_controller.get_available_series()
        all_algorithms_to_calculate_mp = ["stomp", "stump", "scrimp++", "mstump"]
        timestamps = []
        time_series = []
        matrix_profile = []

        try:
            if request.method == "POST":
                if 'n' in request.form and 'k' in request.form:
                    time_series_length = int(request.form["n"])
                    time_series_dim = int(request.form["k"])

                    if time_series_length < 100:
                        raise ValueError("Слишком короткий временной ряд")
                    if time_series_length > 1000000:
                        raise ValueError("Слишком длинный ряд")
                    if time_series_dim < 1:
                        raise ValueError("Ряд должен быть размерности не меньше 1")
                    if time_series_dim > 100:
                        raise ValueError("Ряд не должен быть размерности более 100")

                    timestamps, time_series = data_controller.generate_series(time_series_length, time_series_dim)

                    if "algorithm_for_gen" in request.form and "m_for_gen" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_gen"]
                        window_length = int(request.form["m_for_gen"])

                        if matrix_profile_algorithm not in all_algorithms_to_calculate_mp:
                            raise ValueError("Такого алгоритма вычиселния не существует")
                        if window_length < 1:
                            raise ValueError("Размер окна должен быть больше 1")

                        matrix_profile, execution_time = generate_mp(matrix_profile_algorithm, time_series, window_length)
                elif 'series' in request.form:
                    requested_time_series_to_load = request.form["series"]
                    timestamps, time_series = data_controller.get_series(requested_time_series_to_load)
                    if "algorithm_for_choose" in request.form and "m_for_choose" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_choose"]
                        window_length = int(request.form["m_for_choose"])

                        if matrix_profile_algorithm not in all_algorithms_to_calculate_mp:
                            raise ValueError("Такого алгоритма вычиселния не существует")
                        if window_length < 1:
                            raise ValueError("Размер окна должен быть больше 1")

                        matrix_profile, execution_time = generate_mp(matrix_profile_algorithm, time_series, window_length)
                elif 'file' in request.files:
                    uploaded_file = request.files['file']

                    if uploaded_file:
                        file_stream = io.BytesIO(uploaded_file.read())
                        loaded_time_series = pd.read_csv(file_stream)
                        if loaded_time_series.shape[1] > 2:
                            raise ValueError("Данный ряд не одномерный")
                        timestamps = loaded_time_series.iloc[:, 0]
                        if loaded_time_series.iloc[:, 1].isnull().values.any():
                            raise ValueError("Загруженный ряд содержит пропущенные значения или NaN.")
                        df_filtered = loaded_time_series.drop(loaded_time_series.columns[0], axis=1)
                        time_series = [df_filtered[col].tolist() for col in df_filtered.columns]

                    if "algorithm_for_load" in request.form and "m_for_load" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_load"]
                        window_length = int(request.form["m_for_load"])

                        matrix_profile, execution_time = generate_mp(matrix_profile_algorithm, time_series, window_length)

                plot_html = visualize_time_series_with_matrix_profile(timestamps, time_series, matrix_profile)
                return render_template("index.html", plot_html=plot_html, available_series=available_series)
        except ValueError:
            return render_template("index.html", plot_html=plot_html, available_series=available_series)
        return render_template("index.html", plot_html=plot_html, available_series=available_series)

    @app.route("/ad_analysis", methods=["GET", "POST"])
    def ad_analysis():
        """Страница для анализа алгоритмов Anomaly Detection (AD)"""
        plot_html = None
        available_series = data_controller.get_available_series()
        timestamps = []
        time_series = []
        matrix_profile_algorithm = None
        window_length = None
        threshold = None

        try:
            if request.method == "POST":
                if 'n' in request.form and 'k' in request.form:
                    time_series_length = int(request.form["n"])
                    time_series_dim = int(request.form["k"])

                    if time_series_length < 100:
                        raise ValueError("Слишком короткий временной ряд")
                    if time_series_length > 1000000:
                        raise ValueError("Слишком длинный ряд")
                    if time_series_dim < 1:
                        raise ValueError("Ряд должен быть размерности не меньше 1")
                    if time_series_dim > 100:
                        raise ValueError("Ряд не должен быть размерности более 100")

                    timestamps, time_series = data_controller.generate_series(time_series_length, time_series_dim)

                    if "algorithm_for_gen" in request.form and "threshold_for_gen" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_gen"]
                        threshold = float(request.form["threshold_for_gen"])
                        window_length = int(request.form["m_gen"])
                        # anomaly_indexes = algorithm_applier.apply_algorithm_1d(matrix_profile_algorithm, time_series[0], threshold, window_length)
                elif 'series' in request.form:
                    requested_time_series_to_load = request.form["series"]
                    timestamps, time_series = data_controller.get_series(requested_time_series_to_load)

                    if "algorithm_for_choose" in request.form and "threshold_for_choose" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_choose"]
                        threshold = float(request.form["threshold_for_choose"])
                        window_length = int(request.form["m_choose"])
                        # anomaly_indexes = algorithm_applier.apply_algorithm_1d(matrix_profile_algorithm, time_series[0], threshold, window_length)
                elif 'file' in request.files:
                    uploaded_file = request.files['file']

                    if uploaded_file:
                        file_stream = io.BytesIO(uploaded_file.read())
                        loaded_time_series = pd.read_csv(file_stream)
                        if loaded_time_series.shape[1] > 2:
                            raise ValueError("Данный ряд не одномерный")
                        timestamps = loaded_time_series.iloc[:, 0]
                        if loaded_time_series.iloc[:, 1].isnull().values.any():
                            raise ValueError("Загруженный ряд содержит пропущенные значения или NaN.")
                        df_filtered = loaded_time_series.drop(loaded_time_series.columns[0], axis=1)
                        time_series = [df_filtered[col].tolist() for col in df_filtered.columns]

                    if "algorithm_for_load" in request.form and "threshold_for_load" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_load"]
                        threshold = float(request.form["threshold_for_load"])
                        window_length = int(request.form["m_load"])

                if matrix_profile_algorithm is None:
                    print(1)
                    raise ValueError("Алгоритм вычисления матричного профиля должен быть определен")
                if window_length is None:
                    print(2)
                    raise ValueError("Длина окна матричного профиля должен быть определена")
                if threshold is None:
                    print(3)
                    raise ValueError("Порог должен быть определен")
                if matrix_profile_algorithm not in algorithm_applier.algorithms_for_1d_ad:
                    print(4)
                    raise ValueError("Такого алгоритма вычиселния не существует")
                if window_length < 1:
                    print(5)
                    raise ValueError("Размер окна должен быть больше 1")
                if threshold < 0:
                    print(6)
                    raise ValueError("Порог не может быть меньше 0")
                if threshold > 100:
                    print(7)
                    raise ValueError("Порог не может быть больше 100")
                anomaly_indexes = algorithm_applier.apply_algorithm_1d(matrix_profile_algorithm, time_series[0], threshold, window_length)
                plot_html = visualize_time_series_ad(timestamps, time_series[0], anomaly_indexes)
                return render_template("ad_analysis.html", plot_html=plot_html, available_series=available_series)
        except ValueError:
            return render_template("ad_analysis.html", plot_html=plot_html, available_series=available_series)
        return render_template("ad_analysis.html", plot_html=plot_html, available_series=available_series)

    @app.route("/cpd_analysis", methods=["GET", "POST"])
    def cpd_analysis():
        """Страница для анализа алгоритмов Change Point Detection (CPD)"""
        plot_html = None
        available_series = data_controller.get_available_series()
        timestamps = []
        time_series = []
        number_of_regimes = 2
        window_length = None

        try:
            if request.method == "POST":
                if 'n' in request.form and 'k' in request.form:
                    time_series_length = int(request.form["n"])
                    time_series_dim = int(request.form["k"])

                    if time_series_length < 100:
                        raise ValueError("Слишком короткий временной ряд")
                    if time_series_length > 1000000:
                        raise ValueError("Слишком длинный ряд")
                    if time_series_dim < 1:
                        raise ValueError("Ряд должен быть размерности не меньше 1")
                    if time_series_dim > 100:
                        raise ValueError("Ряд не должен быть размерности более 100")

                    window_length = int(request.form["m_gen"])
                    timestamps, time_series = data_controller.generate_series(time_series_length, time_series_dim)
                elif 'series' in request.form:
                    requested_time_series_to_load = request.form["series"]
                    window_length = int(request.form["m_choose"])
                    timestamps, time_series = data_controller.get_series(requested_time_series_to_load)
                elif 'file' in request.files:
                    uploaded_file = request.files['file']
                    window_length = int(request.form["m_load"])

                    if uploaded_file:
                        file_stream = io.BytesIO(uploaded_file.read())
                        loaded_time_series = pd.read_csv(file_stream)

                        if loaded_time_series.shape[1] > 2:
                            raise ValueError("Данный ряд не одномерный")

                        timestamps = loaded_time_series.iloc[:, 0]

                        if loaded_time_series.iloc[:, 1].isnull().values.any():
                            raise ValueError("Загруженный ряд содержит пропущенные значения или NaN.")

                        df_filtered = loaded_time_series.drop(loaded_time_series.columns[0], axis=1)
                        time_series = [df_filtered[col].tolist() for col in df_filtered.columns]

                if window_length is None:
                    raise ValueError("Длина окна матричного профиля должен быть определена")
                if window_length < 1:
                    raise ValueError("Размер окна должен быть больше 1")

                arc_curve = arc_curve_calculator(time_series[0], window_length, number_of_regimes)
                plot_html = visualize_time_series_with_fluss(timestamps, time_series, arc_curve)
                return render_template("cpd_analysis.html", plot_html=plot_html, available_series=available_series)
        except ValueError:
            return render_template("cpd_analysis.html", plot_html=plot_html, available_series=available_series)
        return render_template("cpd_analysis.html", plot_html=plot_html, available_series=available_series)

    @app.route("/multidim_ad_analysis", methods=["GET", "POST"])
    def multidim_ad_analysis():
        """Страница для анализа алгоритмов Anomaly Detection (AD)"""
        plot_html = None
        available_series = data_controller.get_available_series()
        all_algorithms_for_ad = ["pre_sorting", "post_sorting"]
        timestamps = []
        time_series = []
        matrix_profile_algorithm = None
        threshold = None
        window_length = None

        try:
            if request.method == "POST":
                if 'n' in request.form and 'k' in request.form:
                    time_series_length = int(request.form["n"])
                    time_series_dim = int(request.form["k"])

                    if time_series_length < 100:
                        raise ValueError("Слишком короткий временной ряд")
                    if time_series_length > 1000000:
                        raise ValueError("Слишком длинный ряд")
                    if time_series_dim < 1:
                        raise ValueError("Ряд должен быть размерности не меньше 1")
                    if time_series_dim > 100:
                        raise ValueError("Ряд не должен быть размерности более 100")

                    timestamps, time_series = data_controller.generate_series(time_series_length, time_series_dim)

                    if "algorithm_for_gen" in request.form and "threshold_for_gen" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_gen"]
                        threshold = float(request.form["threshold_for_gen"])
                        window_length = int(request.form["m_gen"])
                elif 'series' in request.form:
                    requested_time_series_to_load = request.form["series"]
                    timestamps, time_series = data_controller.get_series(requested_time_series_to_load)

                    if "algorithm_for_choose" in request.form and "threshold_for_choose" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_choose"]
                        threshold = float(request.form["threshold_for_choose"])
                        window_length = int(request.form["m_choose"])
                elif 'file' in request.files:
                    uploaded_file = request.files['file']

                    if uploaded_file:
                        file_stream = io.BytesIO(uploaded_file.read())
                        loaded_time_series = pd.read_csv(file_stream)
                        timestamps = loaded_time_series.iloc[:, 0]
                        df_filtered = loaded_time_series.drop(loaded_time_series.columns[0], axis=1)

                        for col in loaded_time_series.columns:
                            if col.isnull().values.any():
                                raise ValueError("Загруженный ряд содержит пропущенные значения или NaN.")

                        time_series = [df_filtered[col].tolist() for col in df_filtered.columns]

                    if "algorithm_for_load" in request.form and "threshold_for_load" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_load"]
                        threshold = float(request.form["threshold_for_load"])
                        window_length = int(request.form["m_load"])

                if matrix_profile_algorithm is None:
                    raise ValueError("Алгоритм вычисления матричного профиля должен быть определен")
                if window_length is None:
                    raise ValueError("Длина окна матричного профиля должен быть определена")
                if threshold is None:
                    raise ValueError("Порог должен быть определен")
                if matrix_profile_algorithm not in all_algorithms_for_ad:
                    raise ValueError("Такого алгоритма вычиселния не существует")
                if window_length < 1:
                    raise ValueError("Размер окна должен быть больше 1")
                if threshold < 0:
                    raise ValueError("Порог не может быть меньше 0")
                if threshold > 100:
                    raise ValueError("Порог не может быть больше 100")

                anomaly_indexes, matrix_profile = algorithm_applier.apply_algorithm_nd(matrix_profile_algorithm, time_series, threshold, window_length)
                plot_html = visualize_time_series_ad_multidim(timestamps, time_series, anomaly_indexes, matrix_profile)
                return render_template("multidim_ad_analysis.html", plot_html=plot_html, available_series=available_series)
        except ValueError:
            return render_template("multidim_ad_analysis.html", plot_html=plot_html, available_series=available_series)
        return render_template("multidim_ad_analysis.html", plot_html=plot_html, available_series=available_series)
