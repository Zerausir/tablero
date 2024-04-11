from django_plotly_dash import DjangoDash
from django.conf import settings
import dash
import pandas as pd
from dash import html, dash_table, dcc, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from .services import process_data
from .utils import update_table

# Process the data to get the final DataFrame
df_final = process_data()

app = DjangoDash(
    name='GprApp',
    add_bootstrap_links=True,
    external_stylesheets=["/static/css/inner.css"]
)


# Function to create pie charts for each INDICADOR_CORTO
def create_pie_charts(df):
    grouped = df.groupby('INDICADOR_CORTO')['Nro. INFORME'].count().reset_index()
    grouped['Porcentaje'] = (grouped['Nro. INFORME'] / 50) * 100

    pie_charts = []
    row = []
    for i, (_, row_data) in enumerate(grouped.iterrows()):
        fig = go.Figure(data=[
            go.Pie(labels=['Avance', 'Restante'], values=[row_data['Porcentaje'], 100 - row_data['Porcentaje']],
                   hole=0.3)])
        fig.update_layout(title_text=f"Avance de {row_data['INDICADOR_CORTO']} ({row_data['Nro. INFORME']} informes)",
                          title_x=0.5)
        row.append(dcc.Graph(figure=fig))

        # Cada 3 gráficos, o si es el último gráfico, agregamos la fila actual a pie_charts y comenzamos una nueva fila
        if (i + 1) % 3 == 0 or i == len(grouped) - 1:
            pie_charts.append(html.Div(row, style={'display': 'flex', 'justify-content': 'space-around'}))
            row = []

    return pie_charts


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

    # Div for pie charts
    html.Div(id='pie-charts-container'),
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


# Callback to update pie charts based on the filtered data
@app.callback(
    Output('pie-charts-container', 'children'),
    [Input('table1', 'data')]
)
def update_pie_charts(table_data):
    filtered_df = pd.DataFrame(table_data)
    pie_charts = create_pie_charts(filtered_df)
    return pie_charts


if __name__ == '__main__':
    app.run_server(debug=True)
