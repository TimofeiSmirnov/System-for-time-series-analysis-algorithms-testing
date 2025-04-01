import io
import pandas as pd
from flask import render_template, request, current_app, redirect, url_for
from app.algorithms.apply_algorithms import ApplyAnomalyDetectionAlgorithms
from app.data_controller.data_controller import DataController
from app.utils.mp_calculator import generate_mp, arc_curve_calculator
from app.utils.visualization import visualize_time_series_with_matrix_profile, visualize_time_series_ad, \
    visualize_time_series_with_fluss, visualize_time_series_ad_multidim
from app.test_checker.checker import Checker


def init_routes(app):
    algorithm_applier = ApplyAnomalyDetectionAlgorithms()
    data_controller = DataController()
    checker = Checker()

    @app.errorhandler(404)
    def page_not_found(e):
        return redirect(url_for('index'))

    @app.route("/")
    def index():
        return render_template("main_page.html")

    @app.route("/matrix_profile", methods=["GET", "POST"])
    def main():
        """
        Страница для генерации и визуализации матричного профиля
        """
        plot_html = None
        available_series = data_controller.get_available_series()
        all_algorithms_to_calculate_mp = ["stomp", "stump", "scrimp++", "mstump"]
        timestamps = []
        time_series = []
        matrix_profile = []
        file_name = ""
        error = None

        try:
            if request.method == "POST":
                if 'n' in request.form and 'k' in request.form:
                    time_series_length = int(request.form["n"])
                    time_series_dim = int(request.form["k"])
                    avg_pattern_length = int(request.form["avg_pattern_length"])
                    avg_amplitude = float(request.form["avg_amplitude"])
                    default_variance = float(request.form["default_variance"])
                    variance_pattern_length = int(request.form["variance_pattern_length"])
                    variance_amplitude = float(request.form["variance_amplitude"])

                    if time_series_length < 100:
                        raise ValueError("The time series is too short (should be at least 100).")
                    if time_series_length > 1000000:
                        raise ValueError("The time series is too long (should not exceed 1,000,000).")
                    if time_series_dim < 1:
                        raise ValueError("The series must have at least dimension 1.")
                    if time_series_dim > 100:
                        raise ValueError("The series must not exceed 100 dimensions.")
                    if avg_pattern_length < 10:
                        raise ValueError("Average pattern length must be at least 10.")
                    if avg_pattern_length > time_series_length:
                        raise ValueError(
                            f"Average pattern length cannot exceed the length of the time series ({time_series_length}).")
                    if avg_amplitude <= 0:
                        raise ValueError("Average amplitude must be greater than 0.")
                    if default_variance <= 0:
                        raise ValueError("Default variance must be greater than 0.")
                    if variance_pattern_length < 10:
                        raise ValueError("Variance pattern length must be at least 10.")
                    if variance_pattern_length > time_series_length:
                        raise ValueError(
                            f"Variance pattern length cannot exceed the length of the time series ({time_series_length}).")
                    if variance_amplitude <= 0:
                        raise ValueError("Variance amplitude must be greater than 0.")

                    timestamps, time_series = data_controller.generate_series(
                        time_series_length,
                        time_series_dim,
                        avg_pattern_length,
                        avg_amplitude,
                        default_variance,
                        variance_pattern_length,
                        variance_amplitude
                    )

                    if "algorithm_for_gen" in request.form and "m_for_gen" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_gen"]
                        window_length = int(request.form["m_for_gen"])

                        if matrix_profile_algorithm not in all_algorithms_to_calculate_mp:
                            raise ValueError("The specified algorithm does not exist")
                        if window_length < 1:
                            raise ValueError("Window length must be greater than 1")

                        matrix_profile, execution_time = generate_mp(matrix_profile_algorithm, time_series,
                                                                     window_length)
                elif 'series' in request.form:
                    requested_time_series_to_load = request.form["series"]
                    file_name = requested_time_series_to_load
                    timestamps, time_series = data_controller.get_series(requested_time_series_to_load)
                    if "algorithm_for_choose" in request.form and "m_for_choose" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_choose"]
                        window_length = int(request.form["m_for_choose"])

                        if matrix_profile_algorithm not in all_algorithms_to_calculate_mp:
                            raise ValueError("The specified algorithm does not exist")
                        if window_length < 1:
                            raise ValueError("Window length must be greater than 1")

                        matrix_profile, execution_time = generate_mp(matrix_profile_algorithm, time_series,
                                                                     window_length)
                elif 'file' in request.files:
                    uploaded_file = request.files['file']
                    if uploaded_file:
                        file_stream = io.BytesIO(uploaded_file.read())
                        loaded_time_series = pd.read_csv(file_stream)
                        timestamps = loaded_time_series.iloc[:, 0]
                        for col in loaded_time_series.columns:
                            if loaded_time_series[col].isnull().values.any():
                                raise ValueError("The uploaded series contains missing or NaN values.")
                        df_filtered = loaded_time_series.drop(loaded_time_series.columns[0], axis=1)
                        time_series = [df_filtered[col].tolist() for col in df_filtered.columns]

                    if "algorithm_for_load" in request.form and "m_for_load" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_load"]
                        window_length = int(request.form["m_for_load"])
                        matrix_profile, execution_time = generate_mp(matrix_profile_algorithm, time_series,
                                                                     window_length)

                plot_html = visualize_time_series_with_matrix_profile(timestamps, time_series, matrix_profile,
                                                                      file_name)
                return render_template("index.html", plot_html=plot_html, available_series=available_series,
                                       error=error)
        except Exception as e:
            error = e
            return render_template("index.html", plot_html=plot_html, available_series=available_series, error=error)
        return render_template("index.html", plot_html=plot_html, available_series=available_series, error=error)

    @app.route("/ad_analysis", methods=["GET", "POST"])
    def ad_analysis():
        """
        Страница для анализа алгоритмов Anomaly Detection (AD)
        """
        plot_html = None
        available_series = data_controller.get_available_series()
        timestamps = []
        time_series = []
        algorithm = None
        window_length = None
        threshold = None
        window_length_damp = 300
        learn_length_damp = 400
        file_name = ""
        error = None

        try:
            if request.method == "POST":
                if 'n' in request.form and 'k' in request.form:
                    time_series_length = int(request.form["n"])
                    time_series_dim = int(request.form["k"])
                    avg_pattern_length = int(request.form["avg_pattern_length"])
                    avg_amplitude = float(request.form["avg_amplitude"])
                    default_variance = float(request.form["default_variance"])
                    variance_pattern_length = int(request.form["variance_pattern_length"])
                    variance_amplitude = float(request.form["variance_amplitude"])

                    if time_series_length < 100:
                        raise ValueError("The time series is too short (should be at least 100).")
                    if time_series_length > 1000000:
                        raise ValueError("The time series is too long (should not exceed 1,000,000).")
                    if time_series_dim < 1:
                        raise ValueError("The series must have at least dimension 1.")
                    if time_series_dim > 100:
                        raise ValueError("The series must not exceed 100 dimensions.")
                    if avg_pattern_length < 10:
                        raise ValueError("Average pattern length must be at least 10.")
                    if avg_pattern_length > time_series_length:
                        raise ValueError(
                            f"Average pattern length cannot exceed the length of the time series ({time_series_length}).")
                    if avg_amplitude <= 0:
                        raise ValueError("Average amplitude must be greater than 0.")
                    if default_variance <= 0:
                        raise ValueError("Default variance must be greater than 0.")
                    if variance_pattern_length < 10:
                        raise ValueError("Variance pattern length must be at least 10.")
                    if variance_pattern_length > time_series_length:
                        raise ValueError(
                            f"Variance pattern length cannot exceed the length of the time series ({time_series_length}).")
                    if variance_amplitude <= 0:
                        raise ValueError("Variance amplitude must be greater than 0.")

                    timestamps, time_series = data_controller.generate_series(
                        time_series_length,
                        time_series_dim,
                        avg_pattern_length,
                        avg_amplitude,
                        default_variance,
                        variance_pattern_length,
                        variance_amplitude
                    )

                    if "algorithm_for_gen" in request.form and "threshold_for_gen" in request.form:
                        algorithm = request.form["algorithm_for_gen"]
                        threshold = float(request.form["threshold_for_gen"])
                        window_length = int(request.form["m_gen"])
                        if algorithm == "damp":
                            window_length_damp = int(request.form["window_length_damp_gen"])
                            learn_length_damp = int(request.form["learn_length_damp_gen"])

                        # anomaly_indexes = algorithm_applier.apply_algorithm_1d(matrix_profile_algorithm, time_series[0], threshold, window_length)
                elif 'series' in request.form:
                    requested_time_series_to_load = request.form["series"]
                    file_name = requested_time_series_to_load
                    timestamps, time_series = data_controller.get_series(requested_time_series_to_load)

                    if "algorithm_for_choose" in request.form and "threshold_for_choose" in request.form:
                        algorithm = request.form["algorithm_for_choose"]
                        threshold = float(request.form["threshold_for_choose"])
                        window_length = int(request.form["m_choose"])

                        if algorithm == "damp":
                            window_length_damp = int(request.form["window_length_damp_choose"])
                            learn_length_damp = int(request.form["learn_length_damp_choose"])

                        # anomaly_indexes = algorithm_applier.apply_algorithm_1d(matrix_profile_algorithm, time_series[0], threshold, window_length)
                elif 'file' in request.files:
                    uploaded_file = request.files['file']

                    if uploaded_file:
                        file_stream = io.BytesIO(uploaded_file.read())
                        loaded_time_series = pd.read_csv(file_stream)
                        if loaded_time_series.shape[1] > 2:
                            raise ValueError("The series is not one-dimensional")
                        timestamps = loaded_time_series.iloc[:, 0]
                        for col in loaded_time_series.columns:
                            if loaded_time_series[col].isnull().values.any():
                                raise ValueError("The uploaded series contains missing or NaN values.")
                        df_filtered = loaded_time_series.drop(loaded_time_series.columns[0], axis=1)
                        time_series = [df_filtered[col].tolist() for col in df_filtered.columns]

                    if "algorithm_for_load" in request.form and "threshold_for_load" in request.form:
                        algorithm = request.form["algorithm_for_load"]
                        threshold = float(request.form["threshold_for_load"])
                        window_length = int(request.form["m_load"])

                        if algorithm == "damp":
                            window_length_damp = int(request.form["window_length_damp_load"])
                            learn_length_damp = int(request.form["learn_length_damp_load"])

                if algorithm is None:
                    raise ValueError("The matrix profile algorithm must be specified")
                if window_length is None:
                    raise ValueError("The matrix profile window length must be specified")
                if threshold is None:
                    raise ValueError("The threshold must be specified")
                if algorithm not in algorithm_applier.algorithms_for_1d_ad:
                    raise ValueError("The specified algorithm does not exist")
                if window_length < 1:
                    raise ValueError("Window length must be greater than 1")
                if threshold < 0:
                    raise ValueError("Threshold cannot be less than 0")
                if threshold > 100:
                    raise ValueError("Threshold cannot be greater than 100")
                if window_length_damp > learn_length_damp:
                    raise ValueError("DAMP window length must be less than the learning sample length")

                anomaly_indexes = algorithm_applier.apply_algorithm_1d(algorithm, time_series[0], threshold,
                                                                       window_length,
                                                                       [window_length_damp, learn_length_damp])
                plot_html = visualize_time_series_ad(timestamps, time_series[0], anomaly_indexes, file_name)
                return render_template("ad_analysis.html", plot_html=plot_html, available_series=available_series)
        except Exception as e:
            error = e
            print(e)
            return render_template("ad_analysis.html", plot_html=plot_html, available_series=available_series,
                                   error=error)
        return render_template("ad_analysis.html", plot_html=plot_html, available_series=available_series, error=error)

    @app.route("/cpd_analysis", methods=["GET", "POST"])
    def cpd_analysis():
        """
        Страница для анализа алгоритмов Change Point Detection (CPD)
        """
        plot_html = None
        available_series = data_controller.get_available_series()
        timestamps = []
        time_series = []
        number_of_regimes = 2
        window_length = None
        file_name = ""
        error = None

        try:
            if request.method == "POST":
                if 'n' in request.form and 'k' in request.form:
                    time_series_length = int(request.form["n"])
                    time_series_dim = int(request.form["k"])
                    number_of_regimes = int(request.form["number_of_segments_gen"])
                    avg_pattern_length = int(request.form["avg_pattern_length"])
                    avg_amplitude = float(request.form["avg_amplitude"])
                    default_variance = float(request.form["default_variance"])
                    variance_pattern_length = int(request.form["variance_pattern_length"])
                    variance_amplitude = float(request.form["variance_amplitude"])

                    if time_series_length < 100:
                        raise ValueError("The time series is too short")
                    if time_series_length > 1000000:
                        raise ValueError("The time series is too long")
                    if time_series_dim < 1:
                        raise ValueError("The series must have at least dimension 1")
                    if time_series_dim > 100:
                        raise ValueError("The series must not have more than 100 dimensions")

                    window_length = int(request.form["m_gen"])
                    timestamps, time_series = data_controller.generate_series(
                        time_series_length,
                        time_series_dim,
                        avg_pattern_length,
                        avg_amplitude,
                        default_variance,
                        variance_pattern_length,
                        variance_amplitude
                    )
                elif 'series' in request.form:
                    requested_time_series_to_load = request.form["series"]
                    file_name = requested_time_series_to_load
                    number_of_regimes = int(request.form["number_of_segments_choose"])
                    window_length = int(request.form["m_choose"])
                    timestamps, time_series = data_controller.get_series(requested_time_series_to_load)
                elif 'file' in request.files:
                    uploaded_file = request.files['file']
                    number_of_regimes = int(request.form["number_of_segments_load"])
                    window_length = int(request.form["m_load"])

                    if uploaded_file:
                        file_stream = io.BytesIO(uploaded_file.read())
                        loaded_time_series = pd.read_csv(file_stream)

                        if loaded_time_series.shape[1] > 2:
                            raise ValueError("The series is not univariate")

                        timestamps = loaded_time_series.iloc[:, 0]

                        for col in loaded_time_series.columns:
                            if loaded_time_series[col].isnull().values.any():
                                raise ValueError("The uploaded series contains missing values or NaNs.")

                        df_filtered = loaded_time_series.drop(loaded_time_series.columns[0], axis=1)
                        time_series = [df_filtered[col].tolist() for col in df_filtered.columns]

                if window_length is None:
                    raise ValueError("The window length for the matrix profile must be specified")
                if window_length < 1:
                    raise ValueError("The window size must be greater than 1")

                arc_curve = arc_curve_calculator(time_series[0], window_length, number_of_regimes)
                plot_html = visualize_time_series_with_fluss(timestamps, time_series, arc_curve, file_name)
                return render_template("cpd_analysis.html", plot_html=plot_html, available_series=available_series,
                                       error=error)
        except Exception as e:
            error = e
            return render_template("cpd_analysis.html", plot_html=plot_html, available_series=available_series,
                                   error=error)
        return render_template("cpd_analysis.html", plot_html=plot_html, available_series=available_series, error=error)

    @app.route("/multidim_ad_analysis", methods=["GET", "POST"])
    def multidim_ad_analysis():
        """
        Страница для анализа алгоритмов Anomaly Detection (AD)
        """
        plot_html = None
        available_series = data_controller.get_available_series()
        all_algorithms_for_ad = ["pre_sorting", "post_sorting", "mstump"]
        timestamps = []
        time_series = []
        matrix_profile_algorithm = None
        threshold = None
        window_length = None
        file_name = ""
        error = None

        try:
            if request.method == "POST":
                if 'n' in request.form and 'k' in request.form:
                    time_series_length = int(request.form["n"])
                    time_series_dim = int(request.form["k"])
                    avg_pattern_length = int(request.form["avg_pattern_length"])
                    avg_amplitude = float(request.form["avg_amplitude"])
                    default_variance = float(request.form["default_variance"])
                    variance_pattern_length = int(request.form["variance_pattern_length"])
                    variance_amplitude = float(request.form["variance_amplitude"])

                    if time_series_length < 100:
                        raise ValueError("The time series is too short (should be at least 100).")
                    if time_series_length > 1000000:
                        raise ValueError("The time series is too long (should not exceed 1,000,000).")
                    if time_series_dim < 1:
                        raise ValueError("The series must have at least dimension 1.")
                    if time_series_dim > 100:
                        raise ValueError("The series must not exceed 100 dimensions.")
                    if avg_pattern_length < 10:
                        raise ValueError("Average pattern length must be at least 10.")
                    if avg_pattern_length > time_series_length:
                        raise ValueError(
                            f"Average pattern length cannot exceed the length of the time series ({time_series_length}).")
                    if avg_amplitude <= 0:
                        raise ValueError("Average amplitude must be greater than 0.")
                    if default_variance <= 0:
                        raise ValueError("Default variance must be greater than 0.")
                    if variance_pattern_length < 10:
                        raise ValueError("Variance pattern length must be at least 10.")
                    if variance_pattern_length > time_series_length:
                        raise ValueError(
                            f"Variance pattern length cannot exceed the length of the time series ({time_series_length}).")
                    if variance_amplitude <= 0:
                        raise ValueError("Variance amplitude must be greater than 0.")

                    timestamps, time_series = data_controller.generate_series(
                        time_series_length,
                        time_series_dim,
                        avg_pattern_length,
                        avg_amplitude,
                        default_variance,
                        variance_pattern_length,
                        variance_amplitude
                    )

                    if "algorithm_for_gen" in request.form and "threshold_for_gen" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_gen"]
                        threshold = float(request.form["threshold_for_gen"])
                        window_length = int(request.form["m_gen"])
                elif 'series' in request.form:
                    requested_time_series_to_load = request.form["series"]
                    file_name = requested_time_series_to_load
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
                            if loaded_time_series[col].isnull().values.any():
                                raise ValueError("The uploaded series contains missing values or NaNs.")

                        time_series = [df_filtered[col].tolist() for col in df_filtered.columns]

                    if "algorithm_for_load" in request.form and "threshold_for_load" in request.form:
                        matrix_profile_algorithm = request.form["algorithm_for_load"]
                        threshold = float(request.form["threshold_for_load"])
                        window_length = int(request.form["m_load"])

                if matrix_profile_algorithm is None:
                    raise ValueError("The matrix profile computation algorithm must be specified")
                if window_length is None:
                    raise ValueError("The window length for the matrix profile must be specified")
                if threshold is None:
                    raise ValueError("The threshold must be specified")
                if matrix_profile_algorithm not in all_algorithms_for_ad:
                    raise ValueError("The specified algorithm does not exist")
                if window_length < 1:
                    raise ValueError("The window size must be greater than 1")
                if threshold < 0:
                    raise ValueError("The threshold cannot be less than 0")
                if threshold > 100:
                    raise ValueError("The threshold cannot be greater than 100")

                anomaly_indexes, matrix_profile = algorithm_applier.apply_algorithm_nd(matrix_profile_algorithm,
                                                                                       time_series, threshold,
                                                                                       window_length)
                plot_html = visualize_time_series_ad_multidim(timestamps, time_series, anomaly_indexes, matrix_profile,
                                                              file_name)
                return render_template("multidim_ad_analysis.html", plot_html=plot_html,
                                       available_series=available_series, error=error)
        except Exception as e:
            error = e
            return render_template("multidim_ad_analysis.html", plot_html=plot_html, available_series=available_series,
                                   error=error)
        return render_template("multidim_ad_analysis.html", plot_html=plot_html, available_series=available_series,
                               error=error)

    @app.route("/ad_test", methods=["GET", "POST"])
    def ad_test():
        """
        Страница для тестирования алгоритмов AD
        """
        plot_html = None
        threshold = 99.9
        window_length = 100
        test_results = None
        mean_results = None
        error = None
        anomaly_part_of_window = 0.5

        try:
            if request.method == "POST":
                if "threshold_damp" in request.form:
                    threshold = float(request.form["threshold_damp"])
                    window_length = int(request.form["window_length_damp"])
                    learn_window_length = int(request.form["learn_window_length_damp"])
                    anomaly_part_of_window = float(request.form["anomaly_marks_damp"])
                    mean_results, test_results = checker.check_ad("damp", threshold, window_length, anomaly_part_of_window, learn_window_length)
                elif "threshold_pre" in request.form:
                    threshold = float(request.form["threshold_pre"])
                    window_length = int(request.form["window_length_pre"])
                    anomaly_part_of_window = float(request.form["anomaly_marks_pre"])
                    mean_results, test_results = checker.check_ad("pre_sorting", threshold, window_length, anomaly_part_of_window)
                elif "threshold_post" in request.form:
                    threshold = float(request.form["threshold_post"])
                    window_length = int(request.form["window_length_post"])
                    anomaly_part_of_window = float(request.form["anomaly_marks_post"])
                    mean_results, test_results = checker.check_ad("post_sorting", threshold, window_length, anomaly_part_of_window)
                elif "threshold_mstump" in request.form:
                    threshold = float(request.form["threshold_mstump"])
                    window_length = int(request.form["window_length_mstump"])
                    anomaly_part_of_window = float(request.form["anomaly_marks_mstump"])
                    mean_results, test_results = checker.check_ad("mstump", threshold, window_length, anomaly_part_of_window)
                elif "threshold_dumb" in request.form:
                    threshold = float(request.form["threshold_dumb"])
                    window_length = int(request.form["window_length_dumb"])
                    anomaly_part_of_window = float(request.form["anomaly_marks_dumb"])
                    mean_results, test_results = checker.check_ad("dumb", threshold, window_length, anomaly_part_of_window)
                return render_template("test_ad.html", test_results=test_results, mean_results=mean_results, error=error)
        except Exception as e:
            error = e
            return render_template("test_ad.html", test_results=test_results, mean_results=mean_results, error=error)
        return render_template("test_ad.html", test_results=test_results, mean_results=mean_results, error=error)

    @app.route("/cpd_test", methods=["GET", "POST"])
    def cpd_test():
        """
        Страница для тестирования алгоритмов CPD
        """
        checker = Checker()
        test_results = None
        mean_results = None
        error = None

        try:
            if request.method == "POST":
                window_length = int(request.form["window_length_cpd"])
                mean_results, test_results = checker.check_cpd(window_length)
                return render_template("test_cpd.html", test_results=test_results, mean_results=mean_results, error=error)
        except Exception as e:
            error = e
            return render_template("test_cpd.html", test_results=test_results, mean_results=mean_results, error=error)
        return render_template("test_cpd.html", test_results=test_results, mean_results=mean_results, error=error)