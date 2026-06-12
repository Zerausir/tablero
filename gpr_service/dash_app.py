import io
import pandas as pd
import math
import os

from django_plotly_dash import DjangoDash
from datetime import date, datetime
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .services import process_data, verificables, procesar_mes_con_fecha, actualizar_planificada, pact_data, \
    pac_verificables, load_environment_variables
from .utils import calculate_disabled_days_for_year, create_pie_charts_for_indicators, create_summary_pie_charts

app = DjangoDash(
    name='GprApp',
    add_bootstrap_links=True,
    external_stylesheets=["/static/css/inner.css"]
)

# ── Estilos de sidebar ────────────────────────────────────────────────────────
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

_BTN_PRIMARY = {
    'width': '100%',
    'padding': '10px 0',
    'background': '#854F0B',
    'color': '#fff',
    'border': 'none',
    'borderRadius': '8px',
    'fontSize': '13px',
    'fontWeight': '600',
    'cursor': 'pointer',
    'marginTop': '8px',
}

_BTN_SECONDARY = {
    'width': '100%',
    'padding': '9px 0',
    'background': 'transparent',
    'color': '#854F0B',
    'border': '1px solid #854F0B',
    'borderRadius': '8px',
    'fontSize': '13px',
    'fontWeight': '600',
    'cursor': 'pointer',
    'marginTop': '8px',
}

app.layout = html.Div([

    # ── Sidebar de controles ───────────────────────────────────────────────────
    html.Div([

        html.Div('Año de análisis', style=_SIDEBAR_LABEL_FIRST),
        dcc.Dropdown(
            id='year-selector',
            options=[
                {'label': '2024', 'value': 2024},
                {'label': '2025', 'value': 2025},
            ],
            value=2025,
            clearable=False,
            style={'marginBottom': '4px'},
        ),

        html.Div('Fecha de corte', style=_SIDEBAR_LABEL),
        dcc.DatePickerSingle(
            id='fecha-seleccionada',
            min_date_allowed=date(2024, 1, 1),
            max_date_allowed=date(2025, 12, 31),
            initial_visible_month=date(2025, 1, 1),
            date=date(2025, 1, 31),
            disabled_days=calculate_disabled_days_for_year(2025),
            display_format='DD/MM/YYYY',
            style={'width': '100%', 'marginBottom': '4px'},
        ),

        html.Div('Filtrar por indicador', style=_SIDEBAR_LABEL),
        html.Div([
            dcc.Dropdown(
                id='filter-indicador',
                options=[],
                multi=True,
                placeholder='Todos los indicadores',
            ),
        ], id='filter-indicador-container', style={'marginBottom': '4px'}),

        # Acciones
        html.Div('Acciones', style=_SIDEBAR_LABEL),

        html.Button(
            'Mostrar / Ocultar datos PACT',
            id='toggle-data-btn1',
            n_clicks=0,
            style=_BTN_SECONDARY,
        ),
        html.Button(
            'Mostrar / Ocultar datos CZO2',
            id='toggle-data-btn',
            n_clicks=0,
            style=_BTN_SECONDARY,
        ),
        html.Button(
            'Detalle del indicador',
            id='toggle-detail-btn',
            n_clicks=0,
            style=_BTN_PRIMARY,
        ),

        # Descarga Anexo 3.1
        html.Div([
            html.Button(
                'Descargar Anexo 3.1',
                id='btn_anexo31',
                style={**_BTN_SECONDARY, 'marginTop': '24px'},
            ),
            dcc.Download(id='download-anexo31'),
        ], id='download-container-anexo31'),

    ], style={
        'width': '260px',
        'flexShrink': '0',
        'background': '#ffffff',
        'borderRight': '1px solid rgba(0,0,0,.10)',
        'padding': '20px 18px',
        'overflowY': 'auto',
        'display': 'flex',
        'flexDirection': 'column',
    }),

    # ── Área principal ─────────────────────────────────────────────────────────
    html.Div([

        # Tablas de datos (ocultas por defecto, toggle via callbacks)
        html.Div([
            html.Div(id='datatable-container1'),
            html.Div([
                html.Button('Descargar datos PACT', id='btn_excel1', style={
                    'padding': '8px 18px',
                    'background': '#0B3D91',
                    'color': '#fff',
                    'border': 'none',
                    'borderRadius': '8px',
                    'fontSize': '13px',
                    'cursor': 'pointer',
                    'marginBottom': '12px',
                }),
                dcc.Download(id='download-excel1'),
            ], id='download-container1', style={'display': 'none'}),
        ], id='datatable-wrapper1', style={'display': 'none', 'marginBottom': '8px'}),

        html.Div([
            html.Div(id='datatable-container'),
            html.Div([
                html.Button('Descargar datos CZO2', id='btn_excel', style={
                    'padding': '8px 18px',
                    'background': '#0B3D91',
                    'color': '#fff',
                    'border': 'none',
                    'borderRadius': '8px',
                    'fontSize': '13px',
                    'cursor': 'pointer',
                    'marginBottom': '12px',
                }),
                dcc.Download(id='download-excel'),
            ], id='download-container', style={'display': 'none'}),
        ], id='datatable-wrapper', style={'display': 'none', 'marginBottom': '8px'}),

        # Gráficos de torta / progreso
        html.Div(id='pie-chart-container', style={'marginBottom': '16px'}),

        # Stores
        dcc.Store(id='stored-df-final'),
        dcc.Store(id='stored-df-pact-verificables'),
        dcc.Store(id='stored-df-final-filtered', data={}),
        dcc.Store(id='stored-df-pact-verificables-filtered', data={}),
        dcc.Store(id='stored-selected-year', data=2025),

        # Interval para refresco automático
        dcc.Interval(
            id='interval-component',
            interval=30 * 1000,
            n_intervals=0,
        ),

    ], style={
        'flex': '1',
        'minWidth': '0',
        'padding': '20px',
        'overflowY': 'auto',
        'background': '#f0f2f5',
    }),

], style={
    'display': 'flex',
    'flexDirection': 'row',
    'minHeight': '100%',
    'height': 'auto',
    'background': '#f0f2f5',
})


# ── Callbacks (sin cambios funcionales) ───────────────────────────────────────

@app.callback(
    [
        Output('fecha-seleccionada', 'min_date_allowed'),
        Output('fecha-seleccionada', 'max_date_allowed'),
        Output('fecha-seleccionada', 'initial_visible_month'),
        Output('fecha-seleccionada', 'date'),
        Output('fecha-seleccionada', 'disabled_days'),
        Output('stored-selected-year', 'data'),
    ],
    [Input('year-selector', 'value')]
)
def update_date_picker(selected_year):
    min_date = date(selected_year, 1, 1)
    max_date = date(selected_year, 12, 31)
    initial_date = date(selected_year, 1, 31)
    disabled_days = calculate_disabled_days_for_year(selected_year)
    return min_date, max_date, min_date, initial_date, disabled_days, selected_year


@app.callback(
    Output('filter-indicador', 'options'),
    [Input('stored-df-pact-verificables', 'data')]
)
def update_indicator_dropdown(json_data):
    if json_data:
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')
        indicadores_disponibles = sorted(df['INDICADOR_CORTO'].unique().tolist())
        return [{'label': i, 'value': i} for i in indicadores_disponibles]
    return []


@app.callback(
    [
        Output('datatable-container1', 'children'),
        Output('datatable-container', 'children'),
        Output('stored-df-final', 'data'),
        Output('stored-df-pact-verificables', 'data'),
        Output('stored-df-final-filtered', 'data'),
        Output('stored-df-pact-verificables-filtered', 'data'),
    ],
    [
        Input('fecha-seleccionada', 'date'),
        Input('filter-indicador', 'value'),
        Input('interval-component', 'n_intervals'),
        Input('stored-selected-year', 'data'),
    ]
)
def update_data_on_date_and_indicator_selection(selected_date, selected_indicators, n_intervals, selected_year):
    df_pact = pact_data(selected_date, selected_year)

    if selected_date:
        df_final = process_data(selected_date, selected_year)
        df_verificables = verificables(df_final, selected_date)
        df_pact_actualizado = actualizar_planificada(procesar_mes_con_fecha(df_pact, selected_date))
        df_pact_verificables = pac_verificables(df_pact_actualizado, df_verificables)

    if selected_indicators:
        df_pact_verificables_filtered = df_pact_verificables[
            df_pact_verificables['INDICADOR_CORTO'].isin(selected_indicators)]
        df_final_filtered = df_final[df_final['INDICADOR_CORTO'].isin(selected_indicators)]
    else:
        df_pact_verificables_filtered = df_pact_verificables
        df_final_filtered = df_final

    table1 = create_data_table(df_pact_verificables_filtered)
    table2 = create_data_table(df_final_filtered)

    return [
        table1,
        table2,
        df_final.to_json(date_format='iso', orient='split'),
        df_pact_verificables.to_json(date_format='iso', orient='split'),
        df_final_filtered.to_json(date_format='iso', orient='split'),
        df_pact_verificables_filtered.to_json(date_format='iso', orient='split'),
    ]


def create_data_table(dataframe):
    return html.Div([
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in dataframe.columns],
            data=dataframe.to_dict('records'),
            style_table={'overflowX': 'auto', 'maxHeight': '600px'},
            style_cell={
                'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
                'overflow': 'hidden', 'textOverflow': 'ellipsis',
                'fontSize': '13px',
            },
            style_header={
                'backgroundColor': '#f7f8fa',
                'fontWeight': '700',
                'textAlign': 'center',
                'fontSize': '12px',
                'borderBottom': '1px solid rgba(0,0,0,.10)',
            },
            editable=True,
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            row_deletable=True,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=100,
            fixed_rows={'headers': True},
        )
    ])


@app.callback(
    Output('datatable-wrapper1', 'style'),
    [Input('toggle-data-btn1', 'n_clicks')]
)
def toggle_datatable1_visibility(n_clicks):
    if n_clicks % 2 == 0:
        return {'display': 'none', 'marginBottom': '8px'}
    return {'display': 'block', 'marginBottom': '8px'}


@app.callback(
    Output('datatable-wrapper', 'style'),
    [Input('toggle-data-btn', 'n_clicks')]
)
def toggle_datatable_visibility(n_clicks):
    if n_clicks % 2 == 0:
        return {'display': 'none', 'marginBottom': '8px'}
    return {'display': 'block', 'marginBottom': '8px'}


@app.callback(
    Output('filter-indicador-container', 'style'),
    Input('toggle-detail-btn', 'n_clicks'),
    State('filter-indicador-container', 'style'),
)
def toggle_filter_visibility(n_clicks, current_style):
    if n_clicks % 2 == 0:
        return {'display': 'none', 'marginBottom': '4px'}
    return {'display': 'block', 'marginBottom': '4px'}


@app.callback(
    Output("download-excel", "data"),
    [Input("btn_excel", "n_clicks")],
    [State('stored-df-final-filtered', 'data')],
)
def download_excel(n_clicks, json_data):
    if n_clicks and json_data:
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        return dcc.send_bytes(excel_buffer.getvalue(), "datos_czo2.xlsx")


@app.callback(
    Output("download-excel1", "data"),
    [Input("btn_excel1", "n_clicks")],
    [State('stored-df-pact-verificables-filtered', 'data')],
)
def download_excel1(n_clicks, json_data):
    if n_clicks and json_data:
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        return dcc.send_bytes(excel_buffer.getvalue(), "datos_pact.xlsx")


@app.callback(
    Output("download-anexo31", "data"),
    [Input("btn_anexo31", "n_clicks")],
    [State('stored-df-pact-verificables-filtered', 'data'),
     State('fecha-seleccionada', 'date')],
)
def download_anexo31(n_clicks, json_data, selected_date):
    if n_clicks and json_data:
        buffer = io.StringIO(json_data)
        df = pd.read_json(buffer, orient='split')
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, sheet_name='Anexo 3.1')
        excel_buffer.seek(0)
        return dcc.send_bytes(excel_buffer.getvalue(), f"Anexo_3.1_{selected_date}.xlsx")


@app.callback(
    Output('pie-chart-container', 'children'),
    [Input('stored-df-pact-verificables-filtered', 'data'),
     Input('toggle-detail-btn', 'n_clicks')],
    [State('fecha-seleccionada', 'date')],
)
def update_pie_charts(json_data, n_clicks, selected_date):
    if json_data is None:
        return []

    buffer = io.StringIO(json_data)
    df = pd.read_json(buffer, orient='split')

    if n_clicks % 2 == 0:
        pie_charts = create_summary_pie_charts(df, selected_date)
        return [
            html.Div(dcc.Graph(figure=pie_charts['Global Planificado']),
                     style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(figure=pie_charts['Fecha de Corte']),
                     style={'width': '50%', 'display': 'inline-block'}),
        ]
    else:
        pie_charts = create_pie_charts_for_indicators(df, selected_date)
        children = []
        row_children = []
        for i, (indicador, (pie_global, pie_corte)) in enumerate(pie_charts.items()):
            row_children.append(
                html.Div(dcc.Graph(figure=pie_global), style={'width': '50%', 'display': 'inline-block'}))
            row_children.append(
                html.Div(dcc.Graph(figure=pie_corte), style={'width': '50%', 'display': 'inline-block'}))
            if i % 2 == 1:
                children.append(html.Div(row_children, style={'display': 'flex', 'flexWrap': 'wrap'}))
                row_children = []
        if row_children:
            children.append(html.Div(row_children, style={'display': 'flex', 'flexWrap': 'wrap'}))
        return children


if __name__ == '__main__':
    app.run_server(debug=True)
