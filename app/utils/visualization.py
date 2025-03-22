import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_time_series_with_matrix_profile(timestamps, values, values_mp):
    """Визуализирует многомерный временной ряд и несколько матричных профилей"""
    num_dimensions = len(values)
    num_mp = len(values_mp)
    print(num_dimensions, num_mp)
    total_rows = num_dimensions + num_mp

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f"Измерение {i + 1}" for i in range(num_dimensions)] +
                       [f"Матричный профиль {i + 1}" for i in range(num_mp)]
    )

    for i in range(num_dimensions):
        fig.add_trace(
            go.Scatter(x=timestamps, y=values[i], mode='lines', name=f"Измерение {i + 1}"),
            row=i + 1, col=1
        )

    for i in range(num_mp):
        fig.add_trace(
            go.Scatter(x=timestamps, y=values_mp[i], mode='lines', name=f"Матричный профиль {i + 1}"),
            row=num_dimensions + i + 1, col=1
        )

    fig.update_layout(
        title="Визуализация многомерного временного ряда и матричных профилей",
        height=300 * total_rows,
        showlegend=True,
    )

    return fig.to_html()


def visualize_time_series_ad(timestamps, values, ad_values):
    """Визуализирует одномерный временной ряд и выделяет аномалии красными точками"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            line=dict(color='blue'),
            name="Временной ряд"
        )
    )

    anomaly_x = [timestamps[i] for i in ad_values]
    anomaly_y = [values[i] for i in ad_values]

    fig.add_trace(
        go.Scatter(
            x=anomaly_x,
            y=anomaly_y,
            mode='markers',
            marker=dict(color='red', size=8),
            name="Аномалии"
        )
    )

    fig.update_layout(
        title="Визуализация временного ряда с аномалиями",
        xaxis_title="Время",
        yaxis_title="Значение",
        showlegend=True
    )

    return fig.to_html()


def visualize_time_series_with_fluss(timestamps, values, values_fluss):
    """Визуализирует многомерный временной ряд и несколько арочных кривых"""
    num_dimensions = len(values)
    num_of_arc_curves = len(values_fluss)

    total_rows = num_dimensions + num_of_arc_curves

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f"Измерение {i + 1}" for i in range(num_dimensions)] +
                       [f"Арочная кривая {i + 1}" for i in range(num_of_arc_curves)]
    )

    for i in range(num_dimensions):
        fig.add_trace(
            go.Scatter(x=timestamps, y=values[i], mode='lines', name=f"Измерение {i + 1}"),
            row=i + 1, col=1
        )

    for i in range(num_of_arc_curves):
        fig.add_trace(
            go.Scatter(x=timestamps, y=values_fluss[i], mode='lines', name=f"Арочная кривая {i + 1}"),
            row=num_dimensions + i + 1, col=1
        )

    fig.update_layout(
        title="Визуализация многомерного временного ряда и арочной кривой",
        height=300 * total_rows,
        showlegend=True,
    )

    return fig.to_html()
