import io
import json
import plotly.graph_objs as go
from dash import dcc, html, Input, Output, State, dash_table, no_update
from dash.exceptions import PreventUpdate
from django.conf import settings
from django.http import HttpRequest, QueryDict
from django_plotly_dash import DjangoDash
import pandas as pd
import dash
from datetime import datetime

from .utils import (convert_timestamps_to_strings, create_heatmap_layout, update_heatmap, update_station_plot_fm,
                    update_station_plot_tv, update_station_plot_am, create_heatmap_data, create_dash_datatable,
                    filter_dataframe_by_frequencies, process_warnings_data, create_warnings_tables)
from .services import customize_data, customize_rtv_warnings_data, marks_rtv_warnings_data

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
            value=['auth_suspension'],
            style={'margin': '10px'}
        ),
        dcc.Checklist(
            id='checkbox-warnings',
            options=[
                {'label': 'Advertencias', 'value': 'show_warnings'}
            ],
            value=['show_warnings'],  # Por defecto, las observaciones están visibles
            style={'margin': '10px'}
        ),
        dcc.Checklist(
            id='checkbox-alerts',
            options=[
                {'label': 'Alertas', 'value': 'show_alerts'}
            ],
            value=['show_alerts'],
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
        dcc.Store(id='store-df-warnings'),
        dcc.Store(id='store-df-alerts'),
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
        [Output('store-df-warnings', 'data'),
         Output('store-df-alerts', 'data')],
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('city-dropdown', 'value')]
    )
    def update_warnings_alerts_data(start_date, end_date, city):
        if not all([start_date, end_date, city]):
            return {}, {}

        request = HttpRequest()
        request.GET = QueryDict(f'start_date={start_date}&end_date={end_date}&city={city}')

        try:
            df_warnings_all, df_alerts_all = marks_rtv_warnings_data(request)

            if not df_warnings_all.empty and not df_alerts_all.empty:
                return df_warnings_all.to_dict('records'), df_alerts_all.to_dict('records')
            else:
                return {}, {}
        except Exception as e:
            print(f"Error al procesar datos de advertencias y alertas: {str(e)}")
            return {}, {}


@app.callback(
    Output('station-plots-container-am', 'children'),
    [Input('frequency-dropdown3', 'value'),
     Input('store-df-original3', 'data'),
     Input('checkbox', 'value'),
     Input('checkbox-warnings', 'value'),
     Input('checkbox-alerts', 'value'),
     Input('city-dropdown', 'value')],
    [State('store-df-warnings', 'data'),
     State('store-df-alerts', 'data')]
)
def update_am_station_plot(freq3, data3, autorizations, show_warnings, show_alerts, ciudad, warnings_data, alerts_data):
    if freq3 and data3:
        df_warnings = pd.DataFrame(warnings_data) if warnings_data else None
        df_alerts = pd.DataFrame(alerts_data) if alerts_data else None
        return html.Div(
            update_station_plot_am(freq3, data3, autorizations, show_warnings, show_alerts, ciudad, df_warnings,
                                   df_alerts))
    return html.Div()


@app.callback(
    Output('station-plots-container-fm', 'children'),
    [Input('frequency-dropdown1', 'value'),
     Input('store-df-original1', 'data'),
     Input('checkbox', 'value'),
     Input('checkbox-warnings', 'value'),
     Input('checkbox-alerts', 'value'),
     Input('city-dropdown', 'value')],
    [State('store-df-warnings', 'data'),
     State('store-df-alerts', 'data')]
)
def update_fm_station_plot(freq1, data1, autorizations, show_warnings, show_alerts, ciudad, warnings_data, alerts_data):
    if freq1 and data1:
        df_warnings = pd.DataFrame(warnings_data) if warnings_data else None
        df_alerts = pd.DataFrame(alerts_data) if alerts_data else None
        return html.Div(
            update_station_plot_fm(freq1, data1, autorizations, show_warnings, show_alerts, ciudad, df_warnings,
                                   df_alerts))
    return html.Div()


@app.callback(
    Output('station-plots-container-tv', 'children'),
    [Input('frequency-dropdown2', 'value'),
     Input('store-df-original2', 'data'),
     Input('checkbox', 'value'),
     Input('checkbox-warnings', 'value'),
     Input('checkbox-alerts', 'value'),
     Input('city-dropdown', 'value')],
    [State('store-df-warnings', 'data'),
     State('store-df-alerts', 'data')]
)
def update_tv_station_plot(freq2, data2, autorizations, show_warnings, show_alerts, ciudad, warnings_data, alerts_data):
    if freq2 and data2:
        df_warnings = pd.DataFrame(warnings_data) if warnings_data else None
        df_alerts = pd.DataFrame(alerts_data) if alerts_data else None
        return html.Div(
            update_station_plot_tv(freq2, data2, autorizations, show_warnings, show_alerts, ciudad, df_warnings,
                                   df_alerts))
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
        df_warnings = customize_rtv_warnings_data(request)

        if df_warnings.empty:
            return "No se encontraron datos de advertencias para los parámetros seleccionados."

        df_5_days, df_9_days, df_60_days, df_91_days = process_warnings_data(df_warnings)
        tables = create_warnings_tables(df_5_days, df_9_days, df_60_days, df_91_days)
        return html.Div(tables)
    except Exception as e:
        import traceback
        error_msg = f"Error al procesar los datos de advertencias: {str(e)}\n\n"
        error_msg += traceback.format_exc()
        print(error_msg)
        return error_msg


@app.callback(
    Output('download-excel', 'data'),
    Input('download-excel-button', 'n_clicks'),
    [State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('city-dropdown', 'value')],
    prevent_initial_call=True
)
def download_tables_excel(n_clicks, start_date, end_date, city):
    if not n_clicks or not all([start_date, end_date, city]):
        raise PreventUpdate

    request = HttpRequest()
    request.GET = QueryDict(f'start_date={start_date}&end_date={end_date}&city={city}')

    try:
        df_warnings = customize_rtv_warnings_data(request)

        if df_warnings.empty:
            raise PreventUpdate

        df_5_days, df_9_days, df_60_days, df_91_days = process_warnings_data(df_warnings)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            def prepare_df_for_excel(df, sheet_name):
                if not df.empty:
                    # Asegurarse de que 'Frecuencia (Hz)' sea tratado como texto
                    if 'Frecuencia (Hz)' in df.columns:
                        df['Frecuencia (Hz)'] = df['Frecuencia (Hz)'].astype(str)

                    # Reemplazar NaN con cadenas vacías
                    df = df.fillna('')

                    # Combine inicio and fin columns
                    for col in df.columns:
                        if col.endswith('_inicio'):
                            col_base = col[:-7]
                            df[col_base] = df.apply(
                                lambda row: f"{row[col]} al {row[col_base + '_fin']}"
                                if row[col] != '' and row[col_base + '_fin'] != ''
                                else '',
                                axis=1
                            )
                            df = df.drop(columns=[col, col_base + '_fin'])

                    # Write DataFrame to Excel
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                    # Get workbook and worksheet objects
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]

                    # Define formats
                    header_format = workbook.add_format({
                        'bold': True,
                        'border': 1,
                        'bg_color': '#D9D9D9'  # Gris claro para el encabezado
                    })
                    warning_format = workbook.add_format({
                        'bg_color': '#FFFF99',  # Amarillo tenue para advertencias
                        'border': 1
                    })
                    alert_format = workbook.add_format({
                        'bg_color': '#FFCCCB',  # Rojo tenue para alertas
                        'border': 1
                    })
                    border_format = workbook.add_format({
                        'border': 1
                    })
                    number_format = workbook.add_format({
                        'border': 1,
                        'num_format': '0'  # Formato numérico sin decimales
                    })

                    # Write headers with format
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num, value, header_format)

                    # Apply autofilter
                    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

                    # Apply conditional formatting and borders
                    for row_num in range(1, len(df) + 1):
                        for col_num, col_name in enumerate(df.columns):
                            try:
                                value = df.iloc[row_num - 1, col_num]
                                cell_format = border_format

                                # Seleccionar el formato según el tipo de columna
                                if col_name == 'Frecuencia (Hz)':
                                    cell_format = number_format
                                elif 'Adv.' in col_name:
                                    cell_format = warning_format
                                elif 'Alert.' in col_name:
                                    cell_format = alert_format

                                # Si es la columna de frecuencia y el valor no está vacío
                                if col_name == 'Frecuencia (Hz)' and value != '':
                                    try:
                                        value = float(value)
                                    except:
                                        pass

                                worksheet.write(row_num, col_num, value, cell_format)
                            except Exception as e:
                                worksheet.write(row_num, col_num, '', cell_format)

                    # Adjust column width
                    for col_num, col_name in enumerate(df.columns):
                        max_length = max(
                            df[col_name].astype(str).apply(len).max(),
                            len(col_name)
                        )
                        worksheet.set_column(col_num, col_num, max_length + 2)

                    # Freeze panes to keep header visible
                    worksheet.freeze_panes(1, 0)

            # Write each table to a different sheet
            prepare_df_for_excel(df_5_days, 'Advertencias 5 días')
            prepare_df_for_excel(df_9_days, 'Alertas 9 días')
            prepare_df_for_excel(df_60_days, 'Advertencias 60 días')
            prepare_df_for_excel(df_91_days, 'Alertas 91 días')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'observaciones_{city}_{timestamp}.xlsx'

        return dcc.send_bytes(output.getvalue(), filename)

    except Exception as e:
        print(f"Error generating Excel file: {str(e)}")
        raise PreventUpdate


register_callbacks()

if __name__ == '__main__':
    app.run_server(debug=True)
