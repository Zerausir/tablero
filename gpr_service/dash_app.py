from django_plotly_dash import DjangoDash
from django.conf import settings
import dash
import pandas as pd
from dash import html, dash_table, dcc, callback
from dash.dependencies import Input, Output

from .services import process_data
from .utils import update_table

# Process the data to get the final DataFrame
df_final = process_data()

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

    # DataTable para mostrar los datos
    dash_table.DataTable(
        id='table1',
        columns=[{"name": i, "id": i} for i in df_final.columns],
        data=df_final.to_dict('records'),
        style_table={'overflowX': 'auto', 'maxHeight': '1000px'},
        style_cell={  # Default cell style
            'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_header={  # Styling for the header to ensure it's consistent with the body
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
        fixed_rows={'headers': True},
    ),
    html.Button("Descargar Excel", id="btn_excel"),
    dcc.Download(id="download-excel"),
])


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
def download_excel(n_clicks, table_data):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_excel, filename="datos_descargados.xlsx", index=False)


if __name__ == '__main__':
    app.run_server(debug=True)
