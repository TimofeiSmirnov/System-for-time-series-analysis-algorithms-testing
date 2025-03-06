import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_time_series(timestamps, values, values_mp):
    """Визуализирует временной ряд и матричный профиль"""

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        subplot_titles=("Временной ряд", "Матричный профиль")
    )

    fig.add_trace(
        go.Scatter(x=timestamps, y=values, mode='lines', name="Временной ряд"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=timestamps, y=values_mp, mode='lines', name="Матричный профиль"),
        row=2, col=1
    )

    fig.update_layout(
        title=" ",
        height=700,
        showlegend=True,
        xaxis_title="Время",
        yaxis_title="Значение"
    )

    fig.update_layout(
        yaxis2_title="Матричный профиль"
    )

    return fig.to_html()
