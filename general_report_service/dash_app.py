import json
import plotly.graph_objs as go
from dash import dcc, html, Input, Output, State
from django.conf import settings
from django.http import HttpRequest, QueryDict
from django_plotly_dash import DjangoDash
import pandas as pd
import dash

from .utils import (convert_timestamps_to_strings, create_heatmap_layout, update_heatmap, update_station_plot_fm,
                    update_station_plot_tv, update_station_plot_am, create_heatmap_data, create_dash_datatable,
                    filter_dataframe_by_frequencies, process_warnings_data, create_warnings_tables)
from .services import customize_data, customize_rtv_warnings_data

app = DjangoDash(
    name='GeneralReportApp',
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
                    id='data-container',
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
        dcc.Store(id='store-selected-frequencies1'),
        dcc.Store(id='store-selected-frequencies2'),
        dcc.Store(id='store-selected-frequencies3'),
        html.Div(id='station-plots-container'),
        html.Div([
            html.Button('Mostrar/Ocultar resultados', id='toggle-results-button', n_clicks=0),
            html.Div([
                dcc.Loading(
                    id="loading-table",
                    type="default",
                    children=[html.Div(id='table-container')]
                )
            ])
        ], id='results-container', style={'display': 'none'}),
        dcc.Store(id='current-tab', data='tab-1'),
    ], style={
        'display': 'flex',
        'flex-direction': 'column',
        'justify-content': 'flex-start',
        'align-items': 'stretch',
        'min-height': '100vh',
        'height': 'auto',
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
         Input('city-dropdown', 'value')],
        [State('store-selected-frequencies1', 'data'),
         State('store-selected-frequencies2', 'data'),
         State('store-selected-frequencies3', 'data')]
    )
    def update_content(fecha_inicio, fecha_fin, ciudad, selected_freq1, selected_freq2, selected_freq3):
        if not all([fecha_inicio, fecha_fin, ciudad]):
            return {}, {}, {}, {}, {}, {}, "Selecciona una fecha inicial, una fecha final y una ciudad"

        if fecha_inicio is not None and fecha_fin is not None and ciudad is not None:
            request = HttpRequest()
            request.GET = QueryDict(mutable=True)
            request.GET['start_date'] = fecha_inicio
            request.GET['end_date'] = fecha_fin
            request.GET['city'] = ciudad

            try:
                df_original1, df_original2, df_original3 = customize_data(request)[:3]
                df_original1 = convert_timestamps_to_strings(df_original1)
                df_original2 = convert_timestamps_to_strings(df_original2)
                df_original3 = convert_timestamps_to_strings(df_original3)

                df_clean1, df_clean2, df_clean3 = customize_data(request)[3:]

                if df_original1.empty and df_original2.empty and df_original3.empty:
                    no_data_message = "No data found for the given parameters."
                    tabs_layout = html.Div()
                else:
                    no_data_message = ""
                    tabs_layout = create_heatmap_layout(df_original1, df_original2, df_original3, selected_freq1,
                                                        selected_freq2, selected_freq3)

                return (df_original1.to_dict('records'), df_original2.to_dict('records'),
                        df_original3.to_dict('records'), df_clean1.to_dict('records'),
                        df_clean2.to_dict('records'), df_clean3.to_dict('records'),
                        html.Div([tabs_layout, html.Div(no_data_message)]))
            except Exception as e:
                return {}, {}, {}, {}, {}, {}, f"Error al obtener los datos: {str(e)}"

    @app.callback(
        [Output('store-selected-frequencies1', 'data'),
         Output('store-selected-frequencies2', 'data'),
         Output('store-selected-frequencies3', 'data')],
        [Input('frequency-dropdown1', 'value'),
         Input('frequency-dropdown2', 'value'),
         Input('frequency-dropdown3', 'value')]
    )
    def store_selected_frequencies(freq1, freq2, freq3):
        return freq1, freq2, freq3

    @app.callback(
        Output('frequency-dropdown1', 'value'),
        [Input('store-selected-frequencies1', 'data')]
    )
    def update_dropdown1(stored_freq):
        return stored_freq

    @app.callback(
        Output('frequency-dropdown2', 'value'),
        [Input('store-selected-frequencies2', 'data')]
    )
    def update_dropdown2(stored_freq):
        return stored_freq

    @app.callback(
        Output('frequency-dropdown3', 'value'),
        [Input('store-selected-frequencies3', 'data')]
    )
    def update_dropdown3(stored_freq):
        return stored_freq

    @app.callback(
        Output('results-container', 'style'),
        [Input('current-tab', 'data')]
    )
    def toggle_results_visibility(current_tab):
        if current_tab == 'tab-4':  # 'tab-4' es la pestaña "Estaciones con Advertencia"
            return {'display': 'none'}
        else:
            return {'display': 'block'}

    @app.callback(
        Output('table-container', 'children'),
        [Input('toggle-results-button', 'n_clicks'),
         Input('frequency-dropdown1', 'value'),
         Input('frequency-dropdown2', 'value'),
         Input('frequency-dropdown3', 'value'),
         Input('current-tab', 'data')],
        [State('store-df-original1', 'data'),
         State('store-df-original2', 'data'),
         State('store-df-original3', 'data'),
         State('table-container', 'children')]
    )
    def update_table_visibility(n_clicks, freq1, freq2, freq3, current_tab, data1, data2, data3, current_table):
        if current_tab == 'tab-4':  # No mostrar la tabla en la pestaña "Estaciones con Advertencia"
            return []

        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'toggle-results-button':
            if current_table is None or (isinstance(current_table, list) and len(current_table) == 0):
                # Si la tabla está oculta, la mostramos
                if current_tab == 'tab-1' and data1:
                    df1 = pd.DataFrame(data1)
                    filtered_df1 = filter_dataframe_by_frequencies(df1, freq1) if freq1 else df1
                    return [create_dash_datatable('table1', filtered_df1)]
                elif current_tab == 'tab-2' and data2:
                    df2 = pd.DataFrame(data2)
                    filtered_df2 = filter_dataframe_by_frequencies(df2, freq2) if freq2 else df2
                    return [create_dash_datatable('table2', filtered_df2)]
                elif current_tab == 'tab-3' and data3:
                    df3 = pd.DataFrame(data3)
                    filtered_df3 = filter_dataframe_by_frequencies(df3, freq3) if freq3 else df3
                    return [create_dash_datatable('table3', filtered_df3)]
            else:
                # Si la tabla está visible, la ocultamos
                return []

        # Si se cambian las frecuencias y la tabla está visible, actualizamos la tabla
        elif current_table is not None and len(current_table) > 0:
            if current_tab == 'tab-1' and data1:
                df1 = pd.DataFrame(data1)
                filtered_df1 = filter_dataframe_by_frequencies(df1, freq1) if freq1 else df1
                return [create_dash_datatable('table1', filtered_df1)]
            elif current_tab == 'tab-2' and data2:
                df2 = pd.DataFrame(data2)
                filtered_df2 = filter_dataframe_by_frequencies(df2, freq2) if freq2 else df2
                return [create_dash_datatable('table2', filtered_df2)]
            elif current_tab == 'tab-3' and data3:
                df3 = pd.DataFrame(data3)
                filtered_df3 = filter_dataframe_by_frequencies(df3, freq3) if freq3 else df3
                return [create_dash_datatable('table3', filtered_df3)]

        return dash.no_update

    @app.callback(
        Output('current-tab', 'data'),
        [Input('tabs-container', 'value')]
    )
    def update_current_tab(tab):
        return tab

    @app.callback(
        Output('heatmap1', 'figure'),
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('city-dropdown', 'value')]
    )
    def update_heatmap1(fecha_inicio, fecha_fin, ciudad):
        if not all([fecha_inicio, fecha_fin, ciudad]):
            return go.Figure()

        request = HttpRequest()
        request.GET = QueryDict(mutable=True)
        request.GET['start_date'] = fecha_inicio
        request.GET['end_date'] = fecha_fin
        request.GET['city'] = ciudad

        try:
            df_original1, _, _ = customize_data(request)[:3]
            df_original1 = convert_timestamps_to_strings(df_original1)
            return create_heatmap_data(df_original1)
        except Exception as e:
            return go.Figure()

    @app.callback(
        Output('heatmap2', 'figure'),
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('city-dropdown', 'value')]
    )
    def update_heatmap2(fecha_inicio, fecha_fin, ciudad):
        if not all([fecha_inicio, fecha_fin, ciudad]):
            return go.Figure()

        request = HttpRequest()
        request.GET = QueryDict(mutable=True)
        request.GET['start_date'] = fecha_inicio
        request.GET['end_date'] = fecha_fin
        request.GET['city'] = ciudad

        try:
            _, df_original2, _ = customize_data(request)[:3]
            df_original2 = convert_timestamps_to_strings(df_original2)
            return create_heatmap_data(df_original2)
        except Exception as e:
            return go.Figure()

    @app.callback(
        Output('heatmap3', 'figure'),
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('city-dropdown', 'value')]
    )
    def update_heatmap3(fecha_inicio, fecha_fin, ciudad):
        if not all([fecha_inicio, fecha_fin, ciudad]):
            return go.Figure()

        request = HttpRequest()
        request.GET = QueryDict(mutable=True)
        request.GET['start_date'] = fecha_inicio
        request.GET['end_date'] = fecha_fin
        request.GET['city'] = ciudad

        try:
            _, _, df_original3 = customize_data(request)[:3]
            df_original3 = convert_timestamps_to_strings(df_original3)
            return create_heatmap_data(df_original3)
        except Exception as e:
            return go.Figure()

    @app.callback(
        Output('station-plots-container-fm', 'children'),
        [Input('frequency-dropdown1', 'value'),
         Input('store-df-original1', 'data'),
         Input('checkbox', 'value'),
         Input('city-dropdown', 'value')]
    )
    def update_fm_station_plot(freq1, data1, autorizations, ciudad):
        if freq1 and data1:
            return html.Div(update_station_plot_fm(freq1, data1, autorizations, ciudad))
        return html.Div()

    @app.callback(
        Output('station-plots-container-tv', 'children'),
        [Input('frequency-dropdown2', 'value'),
         Input('store-df-original2', 'data'),
         Input('checkbox', 'value'),
         Input('city-dropdown', 'value')]
    )
    def update_tv_station_plot(freq2, data2, autorizations, ciudad):
        if freq2 and data2:
            return html.Div(update_station_plot_tv(freq2, data2, autorizations, ciudad))
        return html.Div()

    @app.callback(
        Output('station-plots-container-am', 'children'),
        [Input('frequency-dropdown3', 'value'),
         Input('store-df-original3', 'data'),
         Input('checkbox', 'value'),
         Input('city-dropdown', 'value')]
    )
    def update_am_station_plot(freq3, data3, autorizations, ciudad):
        if freq3 and data3:
            return html.Div(update_station_plot_am(freq3, data3, autorizations, ciudad))
        return html.Div()

    @app.callback(
        Output('new-heatmap-container-fm', 'children'),
        [Input('frequency-dropdown1', 'value'),
         Input('store-df-original1', 'data')]
    )
    def update_new_heatmap_fm(selected_frequencies, stored_data):
        if selected_frequencies:
            return dcc.Graph(figure=update_heatmap(selected_frequencies, stored_data))
        return html.Div()

    @app.callback(
        Output('new-heatmap-container-tv', 'children'),
        [Input('frequency-dropdown2', 'value'),
         Input('store-df-original2', 'data')]
    )
    def update_new_heatmap_tv(selected_frequencies, stored_data):
        if selected_frequencies:
            return dcc.Graph(figure=update_heatmap(selected_frequencies, stored_data))
        return html.Div()

    @app.callback(
        Output('new-heatmap-container-am', 'children'),
        [Input('frequency-dropdown3', 'value'),
         Input('store-df-original3', 'data')]
    )
    def update_new_heatmap_am(selected_frequencies, stored_data):
        if selected_frequencies:
            return dcc.Graph(figure=update_heatmap(selected_frequencies, stored_data))
        return html.Div()

    @app.callback(
        Output('warnings-container', 'children'),
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('city-dropdown', 'value')]
    )
    def update_warnings_tables(start_date, end_date, city):
        if not all([start_date, end_date, city]):
            return "Selecciona una fecha inicial, una fecha final y una ciudad"

        request = HttpRequest()
        request.GET = QueryDict(mutable=True)
        request.GET['start_date'] = start_date
        request.GET['end_date'] = end_date
        request.GET['city'] = city

        try:
            print(f"Fetching warnings data for {city} from {start_date} to {end_date}")
            df_warnings = customize_rtv_warnings_data(request)
            print(f"Warnings data shape: {df_warnings.shape}")
            print(f"Warnings data columns: {df_warnings.columns}")
            print(f"Warnings data head:\n{df_warnings.head()}")
            print(f"Warnings data types:\n{df_warnings.dtypes}")

            if df_warnings.empty:
                return "No se encontraron datos de advertencias para los parámetros seleccionados."

            print("Procesando datos de advertencias...")
            df_5_9_days, df_60_days, df_91_days = process_warnings_data(df_warnings)
            print(
                f"Processed data shapes: 5_9_days={df_5_9_days.shape}, 60_days={df_60_days.shape}, 91_days={df_91_days.shape}")

            print("Creando tablas de advertencias...")
            tables = create_warnings_tables(df_5_9_days, df_60_days, df_91_days)
            return html.Div(tables)
        except Exception as e:
            import traceback
            error_msg = f"Error al procesar los datos de advertencias: {str(e)}\n\n"
            error_msg += traceback.format_exc()
            print(error_msg)
            return error_msg


register_callbacks()

if __name__ == '__main__':
    app.run_server(debug=True)
