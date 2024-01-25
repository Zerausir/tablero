import json

from dash import dcc, html, Input, Output
from django.conf import settings
from django_plotly_dash import DjangoDash

from .utils import convert_timestamps_to_strings, create_heatmap_layout, update_table, update_heatmap
from .services import customize_data

app = DjangoDash(
    name='GeneralReportApp',
    add_bootstrap_links=True,
    external_stylesheets=[
        "/static/css/inner.css",
    ]
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
        dcc.Checklist(
            id='checkbox',
            options=[
                {'label': 'Autorizaciones Suspensión/Baja Potencia', 'value': 'auth_suspension'}
            ],
            value=[],
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
    ], style={'height': '100vh'})


app.layout = define_app_layout()


def register_callbacks():
    @app.callback(
        [Output('store-df-original1', 'data'),
         Output('store-df-original2', 'data'),
         Output('store-df-original3', 'data'),
         Output('store-df-original4', 'data'),  # Store for df_clean1
         Output('store-df-original5', 'data'),  # Store for df_clean2
         Output('store-df-original6', 'data'),  # Store for df_clean3
         Output('data-container', 'children')],  # This will contain both tables and heatmaps
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
            df_original1, df_original2, df_original3 = customize_data(selected_options)[:3]
            df_original1 = convert_timestamps_to_strings(df_original1)
            df_original2 = convert_timestamps_to_strings(df_original2)
            df_original3 = convert_timestamps_to_strings(df_original3)

            df_clean1, df_clean2, df_clean3 = customize_data(selected_options)[3:]

            tabs_layout = create_heatmap_layout(df_original1, df_original2, df_original3)

            return df_original1.to_dict('records'), df_original2.to_dict('records'), df_original3.to_dict('records'), \
                df_clean1.to_dict('records'), df_clean2.to_dict('records'), df_clean3.to_dict('records'), tabs_layout

    @app.callback(
        Output('table1', 'data'),
        [Input('frequency-dropdown1', 'value'),
         Input('store-df-original1', 'data')]
    )
    def update_table1(selected_frequencies1, stored_data1):
        return update_table(selected_frequencies1, stored_data1, 'table1')

    @app.callback(
        Output('table2', 'data'),
        [Input('frequency-dropdown2', 'value'),
         Input('store-df-original2', 'data')]
    )
    def update_table2(selected_frequencies2, stored_data2):
        return update_table(selected_frequencies2, stored_data2, 'table2')

    @app.callback(
        Output('table3', 'data'),
        [Input('frequency-dropdown3', 'value'),
         Input('store-df-original3', 'data')]
    )
    def update_table3(selected_frequencies3, stored_data3):
        return update_table(selected_frequencies3, stored_data3, 'table3')

    @app.callback(
        Output('heatmap1', 'figure'),
        [Input('frequency-dropdown1', 'value'),
         Input('store-df-original4', 'data')]
    )
    def update_heatmap1(selected_frequencies, stored_data):
        return update_heatmap(selected_frequencies, stored_data)

    @app.callback(
        Output('heatmap2', 'figure'),
        [Input('frequency-dropdown2', 'value'),
         Input('store-df-original5', 'data')]
    )
    def update_heatmap2(selected_frequencies, stored_data):
        return update_heatmap(selected_frequencies, stored_data)

    @app.callback(
        Output('heatmap3', 'figure'),
        [Input('frequency-dropdown3', 'value'),
         Input('store-df-original6', 'data')]
    )
    def update_heatmap3(selected_frequencies, stored_data):
        return update_heatmap(selected_frequencies, stored_data)


register_callbacks()

if __name__ == '__main__':
    app.run_server(debug=True)
