import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from django_plotly_dash import DjangoDash

app = DjangoDash(
    name='BandOccupationApp',
    add_bootstrap_links=True,
    external_stylesheets=["/static/css/inner.css"]
)


def define_app_layout():
    return html.Div([
        html.H1(children='Hello Dash'),
        html.Div(children='''Dash: A web application framework for Python.'''),
        dcc.Graph(
            id='example-graph',
            figure=px.scatter(x=[1, 2, 3], y=[4, 1, 2], title="Sample Scatter Plot")
        )
    ])


app.layout = define_app_layout()
