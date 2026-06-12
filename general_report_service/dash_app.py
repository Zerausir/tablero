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
from .excel_report_generator import generate_excel_report

app = DjangoDash(
    name='GeneralReportApp',
    add_bootstrap_links=True,
    external_stylesheets=["/static/css/inner.css"]
)

# ── Estilos internos de la barra lateral ──────────────────────────────────────
_SIDEBAR_LABEL = {
    'fontSize': '10px',
    'fontWeight': '700',
    'textTransform': 'uppercase',
    'letterSpacing': '0.09em',
    'color': '#9ca3af',
    'marginBottom': '8px',
    'marginTop': '16px',
}
_SIDEBAR_LABEL_FIRST = {**_SIDEBAR_LABEL, 'marginTop': '0'}

_FIELD_LABEL = {
    'display': 'block',
    'fontSize': '12px',
    'fontWeight': '500',
    'color': '#4b5563',
    'marginBottom': '5px',
}

_CHECK_ITEM = {
    'display': 'flex',
    'alignItems': 'center',
    'gap': '9px',
    'padding': '9px 12px',
    'background': '#f7f8fa',
    'border': '1px solid rgba(0,0,0,.10)',
    'borderRadius': '8px',
    'fontSize': '13px',
    'color': '#374151',
    'cursor': 'pointer',
    'marginBottom': '8px',
}

_BTN_PRIMARY = {
    'width': '100%',
    'padding': '10px 0',
    'background': '#0B3D91',
    'color': '#fff',
    'border': 'none',
    'borderRadius': '8px',
    'fontSize': '13px',
    'fontWeight': '600',
    'cursor': 'pointer',
    'marginTop': '16px',
}

_BTN_SECONDARY = {
    'width': '100%',
    'padding': '9px 0',
    'background': 'transparent',
    'color': '#0B3D91',
    'border': '1px solid #0B3D91',
    'borderRadius': '8px',
    'fontSize': '13px',
    'fontWeight': '600',
    'cursor': 'pointer',
    'marginTop': '8px',
}


def define_app_layout():
    return html.Div([

        # ── Sidebar de filtros ─────────────────────────────────────────────────
        html.Div([

            html.Div('Período de análisis', style=_SIDEBAR_LABEL_FIRST),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date_placeholder_text="Fecha inicial",
                end_date_placeholder_text="Fecha final",
                minimum_nights=0,
                display_format='DD/MM/YYYY',
                style={'width': '100%', 'marginBottom': '12px'},
            ),

            html.Div('Ciudad', style=_SIDEBAR_LABEL),
            dcc.Dropdown(
                id='city-dropdown',
                options=[{'label': c, 'value': c} for c in json.loads(settings.CITIES)],
                placeholder="Selecciona una ciudad",
                style={'marginBottom': '4px'},
            ),

            html.Div('Opciones de visualización', style=_SIDEBAR_LABEL),
            dcc.Checklist(
                id='checkbox',
                options=[{'label': 'Autorizaciones suspensión / baja potencia', 'value': 'auth_suspension'}],
                value=['auth_suspension'],
                inputStyle={'accentColor': '#0B3D91'},
                labelStyle=_CHECK_ITEM,
            ),
            dcc.Checklist(
                id='checkbox-warnings',
                options=[{'label': 'Advertencias de operación', 'value': 'show_warnings'}],
                value=['show_warnings'],
                inputStyle={'accentColor': '#0B3D91'},
                labelStyle=_CHECK_ITEM,
            ),
            dcc.Checklist(
                id='checkbox-alerts',
                options=[{'label': 'Alertas activas', 'value': 'show_alerts'}],
                value=['show_alerts'],
                inputStyle={'accentColor': '#0B3D91'},
                labelStyle=_CHECK_ITEM,
            ),

            # Mostrar/Ocultar resultados tabulares
            html.Div([
                html.Button(
                    'Mostrar / Ocultar resultados',
                    id='toggle-results-button',
                    n_clicks=0,
                    style=_BTN_SECONDARY,
                ),
            ], id='results-container'),

        ], style={
            'width': '270px',
            'flexShrink': '0',
            'background': '#ffffff',
            'borderRight': '1px solid rgba(0,0,0,.10)',
            'padding': '20px 18px',
            'overflowY': 'auto',
            'display': 'flex',
            'flexDirection': 'column',
        }),

        # ── Área principal de contenido ────────────────────────────────────────
        html.Div([

            dcc.Loading(
                id="loading-1",
                type="circle",
                color='#0B3D91',
                children=[
                    html.Div(
                        id='data-container',
                        style={'minHeight': '400px'},
                    )
                ],
            ),

            # Tabla de datos (toggle)
            dcc.Loading(
                id="loading-table",
                type="circle",
                color='#0B3D91',
                children=[html.Div(id='table-container')],
            ),

            html.Div(id='station-plots-container'),

        ], style={
            'flex': '1',
            'minWidth': '0',
            'padding': '20px',
            'overflowY': 'auto',
            'background': '#f0f2f5',
        }),

        # ── Stores (invisibles) ────────────────────────────────────────────────
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
        dcc.Store(id='current-tab', data='tab-1'),

    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'minHeight': '100%',
        'height': 'auto',
        'background': '#f0f2f5',
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
        """Update table visibility based on current tab and button clicks"""
        if current_tab in ['tab-4', 'tab-5']:
            return []

        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'toggle-results-button':
            if current_table is None or (isinstance(current_table, list) and len(current_table) == 0):
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
                return []

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
        Output('station-plots-container', 'children'),
        [Input('frequency-dropdown1', 'value'),
         Input('frequency-dropdown2', 'value'),
         Input('frequency-dropdown3', 'value'),
         Input('checkbox', 'value'),
         Input('checkbox-warnings', 'value'),
         Input('checkbox-alerts', 'value'),
         Input('current-tab', 'data')],
        [State('store-df-original1', 'data'),
         State('store-df-original2', 'data'),
         State('store-df-original3', 'data'),
         State('store-df-warnings', 'data'),
         State('store-df-alerts', 'data'),
         State('city-dropdown', 'value')]
    )
    def update_station_plots(freq1, freq2, freq3, checkbox_value, checkbox_warnings, checkbox_alerts,
                             current_tab, data1, data2, data3, df_warnings_data, df_alerts_data, ciudad):
        autorizations_selected = 'auth_suspension' in (checkbox_value or [])
        show_warnings = 'show_warnings' in (checkbox_warnings or [])
        show_alerts = 'show_alerts' in (checkbox_alerts or [])

        df_warnings = pd.DataFrame(df_warnings_data) if df_warnings_data else pd.DataFrame()
        df_alerts = pd.DataFrame(df_alerts_data) if df_alerts_data else pd.DataFrame()

        if current_tab == 'tab-1' and freq1 and data1:
            return update_station_plot_fm(freq1, data1, autorizations_selected, show_warnings,
                                          show_alerts, ciudad, df_warnings, df_alerts)
        elif current_tab == 'tab-2' and freq2 and data2:
            return update_station_plot_tv(freq2, data2, autorizations_selected, show_warnings,
                                          show_alerts, ciudad, df_warnings, df_alerts)
        elif current_tab == 'tab-3' and freq3 and data3:
            return update_station_plot_am(freq3, data3, autorizations_selected, show_warnings,
                                          show_alerts, ciudad, df_warnings, df_alerts)
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

            warnings_data = process_warnings_data(df_warnings)
            return create_warnings_tables(warnings_data)
        except Exception as e:
            return f"Error al obtener datos de advertencias: {str(e)}"

    @app.callback(
        [Output('download-excel', 'data'),
         Output('download-excel-button', 'children')],
        [Input('download-excel-button', 'n_clicks')],
        [State('date-picker-range', 'start_date'),
         State('date-picker-range', 'end_date'),
         State('city-dropdown', 'value')]
    )
    def download_warnings_excel(n_clicks, start_date, end_date, city):
        if not n_clicks or not all([start_date, end_date, city]):
            return dash.no_update, 'Descargar Tablas en Excel'

        request = HttpRequest()
        request.GET = QueryDict(mutable=True)
        request.GET['start_date'] = start_date
        request.GET['end_date'] = end_date
        request.GET['city'] = city

        try:
            df_warnings = customize_rtv_warnings_data(request)
            marks_data = marks_rtv_warnings_data(request)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_warnings.to_excel(writer, sheet_name='Advertencias', index=False)
                if marks_data is not None and not marks_data.empty:
                    marks_data.to_excel(writer, sheet_name='Marcas', index=False)

            output.seek(0)
            filename = f'Advertencias_RTV_{city}_{start_date}_{end_date}.xlsx'
            return dcc.send_bytes(output.getvalue(), filename), 'Descargar Tablas en Excel'
        except Exception as e:
            return dash.no_update, f'Error: {str(e)}'

    @app.callback(
        [Output('download-report-excel', 'data'),
         Output('report-status-message', 'children')],
        [Input('generate-report-button', 'n_clicks')],
        [State('report-city-dropdown', 'value'),
         State('report-start-date', 'date'),
         State('report-end-date', 'date'),
         State('report-umbral-am', 'value'),
         State('report-umbral-fm', 'value'),
         State('report-umbral-tv', 'value')]
    )
    def generate_and_download_report(n_clicks, ciudad, fecha_inicio, fecha_fin,
                                     umbral_am, umbral_fm, umbral_tv):
        if n_clicks == 0:
            return dash.no_update, ""

        if not all([ciudad, fecha_inicio, fecha_fin]):
            return dash.no_update, "⚠️ Por favor complete todos los campos obligatorios (Ciudad, Fecha Inicio, Fecha Fin)"

        if umbral_am is None:
            umbral_am = 40
        if umbral_fm is None:
            umbral_fm = 30
        if umbral_tv is None:
            umbral_tv = 47

        try:
            fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
            fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')

            if fecha_inicio_dt > fecha_fin_dt:
                return dash.no_update, "⚠️ La fecha de inicio debe ser anterior a la fecha de fin"

            excel_buffer = generate_excel_report(
                ciudad=ciudad,
                fecha_inicio=fecha_inicio_dt,
                fecha_fin=fecha_fin_dt,
                umbral_am=umbral_am,
                umbral_fm=umbral_fm,
                umbral_tv=umbral_tv
            )

            Year1 = fecha_inicio_dt.year
            Year2 = fecha_fin_dt.year
            Mes_inicio = fecha_inicio_dt.strftime('%B')
            Mes_fin = fecha_fin_dt.strftime('%B')

            if Year1 == Year2 and Mes_inicio == Mes_fin:
                filename = f'RTV_Verificación de parámetros_{ciudad}_{Mes_inicio}_{Year1}.xlsx'
            else:
                filename = f'RTV_Verificación de parámetros_{ciudad}_{Mes_inicio}_{Year1}_{Mes_fin}_{Year2}.xlsx'

            return dcc.send_bytes(excel_buffer.getvalue(), filename), "✅ Reporte generado exitosamente"

        except Exception as e:
            error_msg = f"❌ Error al generar el reporte: {str(e)}"
            print(f"Error in generate_and_download_report: {e}")
            import traceback
            traceback.print_exc()
            return dash.no_update, error_msg

    @app.callback(
        Output('report-city-dropdown', 'value'),
        [Input('city-dropdown', 'value')]
    )
    def sync_report_city(city):
        return city

    @app.callback(
        Output('report-start-date', 'date'),
        [Input('date-picker-range', 'start_date')]
    )
    def sync_report_start_date(start_date):
        return start_date

    @app.callback(
        Output('report-end-date', 'date'),
        [Input('date-picker-range', 'end_date')]
    )
    def sync_report_end_date(end_date):
        return end_date


register_callbacks()

if __name__ == '__main__':
    app.run_server(debug=True)
