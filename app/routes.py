from flask import Flask, render_template, request

from app.utils.data_loader import generate_timestamps, load_time_series
from app.utils.mp_calculator import generate_mp
from app.utils.visualization import visualize_time_series

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Главная страница с возможностью выборов параметра ряда для генерации"""
    plot_html = None
    if request.method == "POST":
        time_series_length = int(request.form['n'])
        window_length = int(request.form['m'])
        algorithm = request.form['algorithm']
        task = request.form['task']
        timestamps, time_series = load_time_series(time_series_length)
        mp = generate_mp(algorithm, time_series, window_length)
        plot_html = visualize_time_series(timestamps, time_series[:(time_series_length - window_length + 1)], mp[0])

    return render_template("index.html", plot_html=plot_html)
