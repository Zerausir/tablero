from django_plotly_dash import DjangoDash
from django.conf import settings
import pandas as pd
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output

from .services import process_data
from .utils import update_table, download_excel, create_pie_charts, ccde01, ccde02, ccde03, ccde04, ccde11, ccdh01, \
    ccds01, ccds03, ccds05, ccds08, ccds11, ccds12, ccds13, ccds16, ccds17, ccds18, ccds30, ccds31, ccds32, ccdr04

# Process the data to get the final DataFrame
df_final = process_data()

# Define un diccionario para los calculadores de porcentaje
calculators = {
    'CCDE-01': ccde01,
    'CCDE-02': ccde02,
    'CCDE-03': ccde03,
    'CCDE-04': ccde04,
    'CCDE-11': ccde11,
    'CCDH-01': ccdh01,
    'CCDS-01': ccds01,
    'CCDS-03': ccds03,
    'CCDS-05': ccds05,
    'CCDS-08': ccds08,
    'CCDS-11': ccds11,
    'CCDS-12': ccds12,
    'CCDS-13': ccds13,
    'CCDS-16': ccds16,
    'CCDS-17': ccds17,
    'CCDS-18': ccds18,
    'CCDS-30': ccds30,
    'CCDS-31': ccds31,
    'CCDS-32': ccds32,
    'CCDR-04': ccdr04
    # Continúa según sea necesario
}

app = DjangoDash(
    name='GprApp',
    add_bootstrap_links=True,
    external_stylesheets=["/static/css/inner.css"]
)

app.layout = html.Div(children=[
    # Filtros Dropdown
    html.Div([
        dcc.Dropdown(
            id='filter-indicador',
            options=[{'label': i, 'value': i} for i in settings.INDICADORES_GPR],
            multi=True,
            placeholder='Filtrar por INDICADOR',
        ),
    ], style={'marginBottom': 10, 'marginTop': 10}),

    html.Button("Mostrar/Ocultar Datos", id="toggle-data-btn", n_clicks=0),
    # DataTable para mostrar los datos
    html.Div(id="datatable-container", style={'display': 'none'},
             children=[dash_table.DataTable(id='table1', columns=[{"name": i, "id": i} for i in df_final.columns],
                                            data=df_final.to_dict('records'),
                                            style_table={'overflowX': 'auto', 'maxHeight': '1000px'},
                                            style_cell={  # Default cell style
                                                'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
                                                'overflow': 'hidden',
                                                'textOverflow': 'ellipsis',
                                            },
                                            style_header={
                                                # Styling for the header to ensure it's consistent with the body
                                                'backgroundColor': 'white',
                                                'fontWeight': 'bold',
                                                'textAlign': 'center',
                                                'overflow': 'hidden',
                                                'textOverflow': 'ellipsis',
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
                                            fixed_rows={'headers': True}),
                       html.Button("Descargar Excel", id="btn_excel"),
                       dcc.Download(id="download-excel"),
                       ]
             ),
    # Div for pie charts
    html.Div(id='pie-charts-container', children=create_pie_charts(df_final, calculators)
             )
])


@app.callback(
    Output('datatable-container', 'style'),
    [Input('toggle-data-btn', 'n_clicks')]
)
def toggle_datatable_visibility(n_clicks):
    if n_clicks % 2 == 0:
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@app.callback(
    Output('pie-charts-container', 'children'),
    [Input('table1', 'data'),
     Input('filter-indicador', 'value')]
)
def update_pie_charts(table_data, selected_indicadores):
    # Crea un DataFrame vacío si no hay datos, de lo contrario convierte table_data en DataFrame
    filtered_df = pd.DataFrame(table_data if table_data else [])

    # Si hay indicadores seleccionados, pasa esta selección a create_pie_charts
    return create_pie_charts(filtered_df, calculators, selected_indicadores)


# Callback para actualizar la DataTable basado en los filtros seleccionados
@app.callback(
    Output('table1', 'data'),
    [Input('filter-indicador', 'value')]
)
def update_table1(selected_indicador):
    return update_table(df_final, selected_indicador, 'table1')


# Callback para la descarga del Excel
@app.callback(
    Output("download-excel", "data"),
    [Input("btn_excel", "n_clicks"),
     Input('table1', 'data')],
    prevent_initial_call=True,
)
def download_excel_callback(n_clicks, table_data):
    return download_excel(n_clicks, table_data, 'table1')


# Callback to update pie charts based on the filtered data


if __name__ == '__main__':
    app.run_server(debug=True)
