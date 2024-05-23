import json
import pandas as pd
from dash import dcc, html, Input, Output
from django.conf import settings
from django_plotly_dash import DjangoDash

from .utils import convert_timestamps_to_strings, create_heatmap_layout, create_heatmap_data
from .services import customize_data

app = DjangoDash(
    name='BandOccupationApp',
    add_bootstrap_links=True,
    external_stylesheets=["/static/css/inner.css"]
)


def define_app_layout():
    return html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date_placeholder_text="Fecha inicial",
            end_date_placeholder_text="Fecha final",
            minimum_nights=0,
            style={'margin': '10px'}
        ),
        dcc.Dropdown(
            id='city-dropdown',
            options=[{'label': ciudad, 'value': ciudad} for ciudad in json.loads(settings.CITIES)],
            placeholder="Selecciona una ciudad",
            style={'margin': '10px'}
        ),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=[
                html.Div(
                    id='data-container',  # This is the placeholder for the heatmaps and tables.
                    style={'height': '70vh', 'overflowY': 'auto'}
                )
            ],
            style={'margin': '10px'}
        ),
        dcc.Store(id='store-df-original1'),
        dcc.Store(id='store-df-original2'),
        dcc.Store(id='store-df-original3'),
        dcc.Store(id='store-df-original4'),
        dcc.Store(id='store-df-original5'),
        dcc.Store(id='store-df-original6'),
    ], style={
        'display': 'flex',
        'flex-direction': 'column',
        'justify-content': 'flex-start',  # Aligns children to the start of the container
        'align-items': 'stretch',  # Stretches children to fill the container width
        'min-height': '100vh',  # Ensures at least the height of the viewport
        'height': 'auto',  # Allows the container to grow beyond the viewport height if needed
    })


app.layout = define_app_layout()


def register_callbacks():
    @app.callback(
        [Output('store-df-original1', 'data'),
         Output('store-df-original2', 'data'),
         Output('store-df-original3', 'data'),
         Output('store-df-original4', 'data'),
         Output('store-df-original5', 'data'),
         Output('store-df-original6', 'data'),
         Output('data-container', 'children')],
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('city-dropdown', 'value')]
    )
    def update_content(fecha_inicio, fecha_fin, ciudad):
        if not all([fecha_inicio, fecha_fin, ciudad]):
            return {}, {}, {}, {}, {}, {}, "Selecciona una fecha inicial, una fecha final y una ciudad"

        if fecha_inicio is not None and fecha_fin is not None and ciudad is not None:
            selected_options = {
                'start_date': fecha_inicio,
                'end_date': fecha_fin,
                'city': ciudad,
            }
            data = customize_data(selected_options)
            df_original1, df_original2, df_original3 = data[:3]
            df_clean1, df_clean2, df_clean3 = data[3:]

            df_original1 = convert_timestamps_to_strings(df_original1)
            df_original2 = convert_timestamps_to_strings(df_original2)
            df_original3 = convert_timestamps_to_strings(df_original3)

            tabs_layout = create_heatmap_layout(df_original1, df_original2, df_original3)

            return df_original1.to_dict('records'), df_original2.to_dict('records'), df_original3.to_dict('records'), \
                df_clean1.to_dict('records'), df_clean2.to_dict('records'), df_clean3.to_dict('records'), tabs_layout

    @app.callback(
        Output('heatmap1', 'figure'),
        [Input('store-df-original1', 'data')]
    )
    def update_heatmap1(data):
        df = pd.DataFrame(data)
        return create_heatmap_data(df)

    @app.callback(
        Output('heatmap2', 'figure'),
        [Input('store-df-original2', 'data')]
    )
    def update_heatmap2(data):
        df = pd.DataFrame(data)
        return create_heatmap_data(df)

    @app.callback(
        Output('heatmap3', 'figure'),
        [Input('store-df-original3', 'data')]
    )
    def update_heatmap3(data):
        df = pd.DataFrame(data)
        return create_heatmap_data(df)


register_callbacks()

if __name__ == '__main__':
    app.run_server(debug=True)
