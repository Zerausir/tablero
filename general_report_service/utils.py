import pandas as pd
import datetime
from django.conf import settings
import plotly.graph_objs as go
from dash import dcc, html, dash_table, no_update
from typing import Union

MONTH_TRANSLATIONS = settings.MONTH_TRANSLATIONS


def convert_end_date(date_time: str) -> datetime.datetime:
    """
    Convert end date string to datetime object with time set to '23:59:59'.

    Args:
        date_time (str): End date string.

    Returns:
        datetime.datetime: Corresponding datetime object.
    """
    try:
        datetime_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d')
        datetime_obj = datetime_obj.replace(hour=23, minute=59, second=59)
        return datetime_obj
    except Exception as e:
        # Log the error and raise a custom exception
        raise Exception("Error converting end date: {}".format(e))


def convert_start_date(date_time: str) -> datetime.datetime:
    """
    Convert start date string to datetime object with time set to '00:00:01'.

    Args:
        date_time (str): Start date string.

    Returns:
        datetime.datetime: Corresponding datetime object.
    """
    try:
        datetime_obj = datetime.datetime.strptime(date_time, '%Y-%m-%d')
        datetime_obj = datetime_obj.replace(hour=0, minute=0, second=1)
        return datetime_obj
    except Exception as e:
        # Log the error and raise a custom exception
        raise Exception("Error converting start date: {}".format(e))


def convert_timestamps_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp columns in a DataFrame to strings in the format 'YYYY-MM-DD'.

    Args:
        df (pandas.DataFrame): The DataFrame to convert.

    Returns:
        pandas.DataFrame: The DataFrame with timestamp columns converted to strings.
    """
    date_columns = ['Tiempo', 'Inicio Autorización', 'Fin Autorización']
    date_format = '%Y-%m-%d'  # Replace with your actual date format

    for col in date_columns:
        if col in df.columns:
            # Convert column to datetime with specified format, coercing errors to NaT
            df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
            # Format datetime as string, ignoring NaT (which become NaT again)
            df[col] = df[col].dt.strftime(date_format).replace('NaT', '-')

    # Handle MultiIndex for columns and index, if applicable
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(
            lambda x: [(y.strftime(date_format) if isinstance(y, pd.Timestamp) else y) for y in x])
    else:
        df.columns = [x.strftime(date_format) if isinstance(x, pd.Timestamp) else x for x in df.columns]

    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.map(lambda x: [(y.strftime(date_format) if isinstance(y, pd.Timestamp) else y) for y in x])
    else:
        df.index = [x.strftime(date_format) if isinstance(x, pd.Timestamp) else x for x in df.index]

    return df


def create_dash_datatable(table_id: str, dataframe: pd.DataFrame) -> dash_table.DataTable:
    """
    Create a Dash DataTable from a Pandas DataFrame with certain pre-defined styles and functionalities.

    Args:
        table_id (str): Identifier for the Dash DataTable.
        dataframe (pd.DataFrame): The DataFrame from which the Dash DataTable is to be created.

    Returns:
        dash_table.DataTable: A Dash DataTable component.
    """
    # Define the range for your color scale
    zmin = 0
    zmax = 130

    # Define the Plotly rainbow color scale
    plotly_rainbow_scale = ['rgb(150,0,90)', 'rgb(0,0,200)', 'rgb(0,25,255)', 'rgb(0,152,255)',
                            'rgb(44,255,150)', 'rgb(151,255,0)', 'rgb(255,234,0)', 'rgb(255,111,0)', 'rgb(255,0,0)']

    # Function to determine if text color should be white or black based on background color
    def text_color_based_on_background(background_color):
        # Parse the background color string and get R, G, B values
        r, g, b = [int(x) for x in background_color.replace('rgb(', '').replace(')', '').split(',')]
        # Calculate the luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        # Return 'white' for dark background colors and 'black' for light background colors
        return 'white' if luminance < 150 else 'black'

    # Create conditional formatting for the 'Level (dBµV/m)' column
    style_data_conditional = [
        {
            'if': {
                'filter_query': f'{{Level (dBµV/m)}} >= {zmin + (zmax - zmin) * i / (len(plotly_rainbow_scale) - 1)} && {{Level (dBµV/m)}} < {zmin + (zmax - zmin) * (i + 1) / (len(plotly_rainbow_scale) - 1)}',
                'column_id': 'Level (dBµV/m)'
            },
            'backgroundColor': plotly_rainbow_scale[i],
            'color': text_color_based_on_background(plotly_rainbow_scale[i])
        }
        for i in range(len(plotly_rainbow_scale))  # Iterate through the color scale
    ]

    return dash_table.DataTable(
        data=dataframe.to_dict('records'),
        columns=[{"name": col, "id": col} for col in dataframe.columns],
        style_table={'overflowX': 'auto', 'maxHeight': '300px'},
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
        id=table_id,
        style_data_conditional=style_data_conditional,
        fixed_rows={'headers': True},  # This line fixes the header row at the top
    )


def create_frequency_dropdown(dropdown_id, dataframe, placeholder_text, selected_frequencies=None):
    if not dataframe.empty and 'Frecuencia (Hz)' in dataframe.columns and 'Estación' in dataframe.columns:
        # Group by frequency and station to get unique combinations
        grouped = dataframe.groupby(['Frecuencia (Hz)', 'Estación']).size().reset_index(name='count')

        # Create options list with both frequency and station information
        options = []
        for _, row in grouped.iterrows():
            freq = row['Frecuencia (Hz)']
            station = row['Estación']
            if station == "-":
                label = f"{freq} Hz"
            else:
                label = f"{freq} Hz | {station}"
            options.append({'label': label, 'value': freq})

        # Sort options by frequency
        options.sort(key=lambda x: x['value'])
    else:
        options = []
        placeholder_text = "No existe información disponible"

    return dcc.Dropdown(
        id=dropdown_id,
        options=options,
        value=selected_frequencies,  # Set the initial value
        placeholder=placeholder_text,
        multi=True,
        style={'margin': '10px'}
    )


def create_heatmap_data(df: pd.DataFrame, selected_frequencies=None) -> dict:
    """
    Create heatmap data from a DataFrame. The DataFrame is expected to have 'Level (dBµV/m)',
    'Tiempo', and 'Frecuencia (Hz)' columns. If the DataFrame is empty, an empty figure is returned.

    Args:
        df (pd.DataFrame): The DataFrame from which the heatmap data is to be created.

    Returns:
        dict: A dictionary containing 'data' and 'layout' for the heatmap.
    """
    if df.empty:
        return go.Figure()

    df = df.fillna(0)

    heatmap_data = df.pivot_table(values='Level (dBµV/m)', index='Tiempo', columns=['Frecuencia (Hz)', 'Estación'])

    # Create the x-axis labels. If 'Estación' is "-", only use 'Frecuencia (Hz)'
    freq_categories = [
        "{} Hz".format(freq) if station == "-" else "{} Hz | {}".format(freq, station)
        for freq, station in heatmap_data.columns
    ]

    if selected_frequencies:
        bottom_margin = 100  # Smaller margin if frequencies are selected
        plot_height = 600  # Smaller height if frequencies are selected
    else:
        bottom_margin = 400  # Larger margin to accommodate x-axis labels
        plot_height = 900  # Larger height for initial view

    return {
        'data': [go.Heatmap(
            z=heatmap_data.values[::-1],
            x=freq_categories,
            y=heatmap_data.index[::-1],
            colorscale='rainbow',
            zmin=0,
            zmax=130,
            colorbar=dict(title='Nivel (dBµV/m)')
        )],
        'layout': go.Layout(
            title='Nivel de Intensidad de Campo Eléctrico por Frecuencia',
            xaxis={
                'title': 'Frecuencia (Hz)',
                'type': 'category',
                'tickvals': list(range(len(freq_categories))),
                'ticktext': freq_categories
            },
            yaxis={'title': 'Tiempo'},
            margin=go.layout.Margin(
                l=100,
                r=100,
                b=bottom_margin,
                t=100,
            ),
            autosize=True,
            height=plot_height,
        )
    }


def create_heatmap_layout(df_original1, df_original2, df_original3, selected_freq1=None, selected_freq2=None,
                          selected_freq3=None):
    dropdown1 = create_frequency_dropdown('frequency-dropdown1', df_original1, "Seleccione una Frecuencia",
                                          selected_freq1)
    dropdown2 = create_frequency_dropdown('frequency-dropdown2', df_original2, "Seleccione una Frecuencia",
                                          selected_freq2)
    dropdown3 = create_frequency_dropdown('frequency-dropdown3', df_original3, "Seleccione una Frecuencia",
                                          selected_freq3)

    tabs_layout = dcc.Tabs(id='tabs-container', children=[
        dcc.Tab(label='Radiodifusión FM', value='tab-1', children=[
            dcc.Graph(id='heatmap1'),
            html.Div(dropdown1, style={'marginBottom': '10px'}),
            html.Div(id='new-heatmap-container-fm'),
            html.Div(id='station-plots-container-fm'),
        ]),
        dcc.Tab(label='Televisión', value='tab-2', children=[
            dcc.Graph(id='heatmap2'),
            html.Div(dropdown2, style={'marginBottom': '10px'}),
            html.Div(id='new-heatmap-container-tv'),
            html.Div(id='station-plots-container-tv'),
        ]),
        dcc.Tab(label='Radiodifusión AM', value='tab-3', children=[
            dcc.Graph(id='heatmap3'),
            html.Div(dropdown3, style={'marginBottom': '10px'}),
            html.Div(id='new-heatmap-container-am'),
            html.Div(id='station-plots-container-am'),
        ]),
    ])

    return tabs_layout


def filter_dataframe_by_frequencies(df: pd.DataFrame, selected_frequencies: list) -> pd.DataFrame:
    """
    Filter the DataFrame based on a list of selected frequencies.

    Args:
        df (pd.DataFrame): The original DataFrame.
        selected_frequencies (list): List of frequencies to filter by.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if selected_frequencies:
        # Convert selected frequencies to float if they are not already.
        selected_frequencies = [float(freq) for freq in selected_frequencies]
        # Filter the DataFrame.
        return df[df['Frecuencia (Hz)'].isin(selected_frequencies)]
    else:
        # If no frequencies are selected, return the DataFrame as is.
        return df


def update_heatmap(selected_frequencies: list, stored_data: list) -> Union[go.Figure, dict]:
    """
    Update heatmap based on the selected frequencies and stored data.

    Args:
        selected_frequencies (list): List of selected frequencies.
        stored_data (list): Previously stored data as a list of dictionaries.

    Returns:
        Union[go.Figure, dict]: Updated heatmap figure or a dictionary containing heatmap data and layout if not empty.
    """
    if stored_data is None:
        return go.Figure()

    df_clean = pd.DataFrame.from_records(stored_data)

    if not selected_frequencies:
        # If no frequencies are selected, use the entire dataset
        filtered_df = df_clean
    else:
        # If frequencies are selected, filter the DataFrame
        filtered_df = df_clean[df_clean['Frecuencia (Hz)'].isin(selected_frequencies)]

    # Generate the heatmap figure using the filtered DataFrame
    return create_heatmap_data(filtered_df)


def update_station_plot_am(selected_frequencies: list, stored_data: list, autorizations_selected: bool,
                           ciudad: str) -> dcc.Graph:
    """
    Update station plot for AM based on selected frequencies, stored data, and authorization status.

    This function generates a series of interactive plots for AM frequencies, showing the daily electric field levels
    and highlighting areas based on different criteria such as authorization status and level thresholds.

    Args:
        selected_frequencies (list): List of selected frequencies to plot.
        stored_data (list): List of dictionaries containing stored data for plotting.
        autorizations_selected (bool): Flag indicating whether to include data related to authorization status in plots.
        ciudad (str): Name of the city for which the plot is being generated.

    Returns:
        dcc.Graph: A Dash core component Graph that contains the interactive plot.
    """
    if not selected_frequencies or not stored_data:
        return no_update

    # Convert stored_data to DataFrame
    df = pd.DataFrame.from_records(stored_data)
    df['Tiempo'] = pd.to_datetime(df['Tiempo'], errors='coerce')
    df = df.sort_values(by='Tiempo', ascending=True)
    df['Tiempo'] = df['Tiempo'].dt.strftime('%Y-%m-%d')
    # Container for the plots
    plots = []

    # Create a plot for each selected frequency
    for frequency in selected_frequencies:
        df_filtered = df[df['Frecuencia (Hz)'] == frequency]
        df_filtered = df_filtered.rename(
            columns={'Frecuencia (Hz)': 'freq', 'Estación': 'est', 'Level (dBµV/m)': 'level',
                     'Bandwidth (Hz)': 'bandwidth', 'Inicio Autorización': 'Fecha_inicio',
                     'Fin Autorización': 'Fecha_fin'})
        df_filtered['Fecha_inicio'] = df_filtered['Fecha_inicio'].fillna(0)
        df_filtered['Fecha_fin'] = df_filtered['Fecha_fin'].fillna(0)
        nombre = df_filtered['est'].iloc[0]

        def minus(row):
            try:
                level = float(row['level'])  # Convert level to float
            except ValueError:
                return 0  # Return 0 if conversion fails

            if level > 0 and level < 40:
                return level
            return 0

        def bet(row):
            try:
                level = float(row['level'])  # Convert level to float
            except ValueError:
                return 0  # Return 0 if conversion fails

            if level >= 40 and level < 62:
                return level
            return 0

        def plus(row):
            try:
                level = float(row['level'])  # Convert level to float
            except ValueError:
                return 0  # Return 0 if conversion fails

            if level >= 62:
                return level
            return 0

        def valor(row):
            """function to return a specific value if the value in every row of the column 'level' meet the
            condition"""
            if row['level'] == 0:
                return 120
            return 0

        def aut(row):
            """function to return a specific value if the value in every row of the column 'level' meet the
            condition"""
            if row['Fecha_fin'] != 0 and row['level'] == 0:
                return 0
            elif row['Fecha_fin'] != 0 and row['level'] != 0:
                return row['level']
            return 0

        """create a new column in the df_filtered frame for every definition (minus, bet, plus, valor, aut)"""
        df_filtered['minus'] = df_filtered.apply(lambda row: minus(row), axis=1)
        df_filtered['bet'] = df_filtered.apply(lambda row: bet(row), axis=1)
        df_filtered['plus'] = df_filtered.apply(lambda row: plus(row), axis=1)
        df_filtered['valor'] = df_filtered.apply(lambda row: valor(row), axis=1)
        df_filtered['aut'] = df_filtered.apply(lambda row: aut(row), axis=1)
        # Creating the plot
        fig = go.Figure()

        colors = {
            'Plus': '#7fc97f',  # Example color, similar to 'Accent'
            'Bet': '#FFD700',  # Example color, similar to 'Set3_r'
            'Minus': '#ff9999',  # Example color, similar to 'Pastel1'
            'Valor': '#beaed4',  # Example color, similar to 'Paired'
            'Autorizaciones': '#386cb0'  # Example color, similar to 'Set2_r'
        }

        # Adding area plots with custom colors
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['plus'], fill='tozeroy',
                                 name='Los valores de campo eléctrico diario superan el valor del borde de área de cobertura (>=62 dBuV/m).',
                                 line=dict(color=colors['Plus'])))
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['bet'], fill='tozeroy',
                                 name='Los valores de campo eléctrico diario se encuentran entre el valor del borde de área de protección y el valor del borde de área de cobertura (entre 40 y 62 dBuV/m).',
                                 line=dict(color=colors['Bet'])))
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['minus'], fill='tozeroy',
                                 name='Los valores de campo eléctrico diario son inferiores al valor del borde de área de protección (<40 dBuV/m).',
                                 line=dict(color=colors['Minus'])))
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['valor'], fill='tozeroy',
                                 name='No se dispone de mediciones del sistema SACER.',
                                 line=dict(color=colors['Valor'])))

        # Use the autorizations_selected flag to determine whether to plot 'autorizaciones' data
        if autorizations_selected:
            fig.add_trace(
                go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['aut'], fill='tozeroy',
                           name='Dispone de autorización para suspensión de emisiones y/o operación con baja potencia.',
                           line=dict(color=colors['Autorizaciones'])))

        # Setting plot layout
        tick_labels = df_filtered['Tiempo'].unique().tolist()
        tick_positions = list(range(len(tick_labels)))  # Convert range to list

        fig.update_layout(
            title=f'Ciudad: {ciudad}, Estación: {nombre}, Frecuencia: {frequency} Hz',
            xaxis=dict(
                title='Tiempo',
                type='category',
                tickangle=-45,
                tickmode='auto',
                nticks=31,
                ticktext=tick_labels,
                tickvals=tick_positions
            ),
            yaxis=dict(
                title='Nivel de Intensidad de Campo Eléctrico (dBµV/m)',
                range=[0, 120]
            ),
            margin=dict(l=100, r=100, t=100, b=100),
            hovermode='closest',
            legend=dict(
                x=0.5,
                y=-0.3,
                traceorder='normal',
                font=dict(
                    size=12,
                ),
                orientation='h',
                xanchor='center',
                yanchor='top'
            )
        )

        # Adding annotations for initial and final dates of the authorization
        if autorizations_selected:
            for mark_time in df_filtered.Fecha_inicio.unique():
                if mark_time and mark_time != 0:  # Ensure this is the correct condition
                    try:
                        mark_index = tick_labels.index(mark_time)
                        fig.add_annotation(
                            x=mark_index, y=0,
                            text=f'Inicio: {mark_time}',
                            showarrow=True,
                            arrowhead=1,
                            ax=0, ay=-60,  # Arrow direction
                            bgcolor="white",  # Background color of the text box
                            bordercolor="black",  # Border color of the text box
                            font=dict(color="black")  # Text font color
                        )
                    except ValueError:
                        pass  # If mark_time is not in tick_labels, skip

            for mark_time in df_filtered.Fecha_fin.unique():
                if mark_time and mark_time != 0:  # Ensure this is the correct condition
                    try:
                        mark_index = tick_labels.index(mark_time)
                        fig.add_annotation(
                            x=mark_index, y=0,
                            text=f'Fin: {mark_time}',
                            showarrow=True,
                            arrowhead=1,
                            ax=0, ay=-30,  # Arrow direction
                            bgcolor="white",  # Background color of the text box
                            bordercolor="black",  # Border color of the text box
                            font=dict(color="black")  # Text font color
                        )
                    except ValueError:
                        pass  # If mark_time is not in tick_labels, skip

        plots.append(dcc.Graph(figure=fig))

    # Return a Div containing all the plots
    return html.Div(plots)


def update_station_plot_fm(selected_frequencies: list, stored_data: list, autorizations_selected: bool,
                           ciudad: str) -> dcc.Graph:
    """
    Update station plot for FM based on selected frequencies, stored data, and authorization status.

    This function generates a series of interactive plots for FM frequencies, showing the daily electric field levels
    and highlighting areas based on different criteria such as authorization status and level thresholds.

    Args:
        selected_frequencies (list): List of selected frequencies to plot.
        stored_data (list): List of dictionaries containing stored data for plotting.
        autorizations_selected (bool): Flag indicating whether to include data related to authorization status in plots.
        ciudad (str): Name of the city for which the plot is being generated.

    Returns:
        dcc.Graph: A Dash core component Graph that contains the interactive plot.
    """
    if not selected_frequencies or not stored_data:
        return no_update

    # Convert stored_data to DataFrame
    df = pd.DataFrame.from_records(stored_data)
    df['Tiempo'] = pd.to_datetime(df['Tiempo'], errors='coerce')
    df = df.sort_values(by='Tiempo', ascending=True)
    df['Tiempo'] = df['Tiempo'].dt.strftime('%Y-%m-%d')
    # Container for the plots
    plots = []

    # Create a plot for each selected frequency
    for frequency in selected_frequencies:
        df_filtered = df[df['Frecuencia (Hz)'] == frequency]
        df_filtered = df_filtered.rename(
            columns={'Frecuencia (Hz)': 'freq', 'Estación': 'est', 'Potencia': 'pot', 'BW Asignado': 'bw',
                     'Level (dBµV/m)': 'level', 'Bandwidth (Hz)': 'bandwidth', 'Inicio Autorización': 'Fecha_inicio',
                     'Fin Autorización': 'Fecha_fin'})
        df_filtered['Fecha_inicio'] = df_filtered['Fecha_inicio'].fillna(0)
        df_filtered['Fecha_fin'] = df_filtered['Fecha_fin'].fillna(0)
        nombre = df_filtered['est'].iloc[0]
        pot = df_filtered['pot'].iloc[0]
        bw = df_filtered['bw'].iloc[0]
        pot = 0 if df_filtered['pot'].iloc[0] == '-' else 1
        bw = 220 if df_filtered['bw'].iloc[0] == '-' else int(df_filtered['bw'].iloc[0])

        def minus(row):
            try:
                level = float(row['level'])  # Convert level to float
            except ValueError:
                return 0  # Return 0 if conversion fails

            if level > 0 and level < 30:
                return level
            return 0

        def bet(row):
            try:
                level = float(row['level'])  # Convert level to float
            except ValueError:
                return 0  # Return 0 if conversion fails

            if pot == 0 and bw == 220:
                if level >= 30 and level < 54:
                    return level
                return 0
            elif pot == 0 and bw == 200:
                if level >= 30 and level < 54:
                    return level
                return 0
            elif pot == 0 and bw == 180:
                if level >= 30 and level < 48:
                    return level
            elif pot == 1:
                if level >= 30 and level < 43:
                    return level
                return 0
            return 0

        def plus(row):
            try:
                level = float(row['level'])  # Convert level to float
            except ValueError:
                return 0  # Return 0 if conversion fails

            if pot == 0 and bw == 220:
                if level >= 54:
                    return level
                return 0
            elif pot == 0 and bw == 200:
                if level >= 54:
                    return level
                return 0
            elif pot == 0 and bw == 180:
                if level >= 48:
                    return level
            elif pot == 1:
                if level >= 43:
                    return level
                return 0
            return 0

        def valor(row):
            """function to return a specific value if the value in every row of the column 'level' meet the
            condition"""
            if row['level'] == 0:
                return 120
            return 0

        def aut(row):
            """function to return a specific value if the value in every row of the column 'level' meet the
            condition"""
            if row['Fecha_fin'] != 0 and row['level'] == 0:
                return 0
            elif row['Fecha_fin'] != 0 and row['level'] != 0:
                return row['level']
            return 0

        """create a new column in the df_filtered frame for every definition (minus, bet, plus, valor, aut)"""
        df_filtered['minus'] = df_filtered.apply(lambda row: minus(row), axis=1)
        df_filtered['bet'] = df_filtered.apply(lambda row: bet(row), axis=1)
        df_filtered['plus'] = df_filtered.apply(lambda row: plus(row), axis=1)
        df_filtered['valor'] = df_filtered.apply(lambda row: valor(row), axis=1)
        df_filtered['aut'] = df_filtered.apply(lambda row: aut(row), axis=1)
        # Creating the plot
        fig = go.Figure()

        colors = {
            'Plus': '#7fc97f',  # Example color, similar to 'Accent'
            'Bet': '#FFD700',  # Example color, similar to 'Set3_r'
            'Minus': '#ff9999',  # Example color, similar to 'Pastel1'
            'Valor': '#beaed4',  # Example color, similar to 'Paired'
            'Autorizaciones': '#386cb0'  # Example color, similar to 'Set2_r'
        }

        if pot == 0 and bw == 220:
            tipo = 'Estereofónico'
            maximo = 54
        elif pot == 0 and bw == 200:
            tipo = 'Estereofónico'
            maximo = 54
        elif pot == 0 and bw == 180:
            tipo = 'Monofónico'
            maximo = 48
        elif pot == 1:
            tipo = 'Baja Potencia'
            maximo = 43
        minimo = 30

        # Adding area plots with custom colors
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['plus'], fill='tozeroy',
                                 name=f'Los valores de campo eléctrico diario superan el valor del borde de área de cobertura ({tipo}: >={maximo} dBuV/m).',
                                 line=dict(color=colors['Plus'])))
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['bet'], fill='tozeroy',
                                 name=f'Los valores de campo eléctrico diario se encuentran entre el valor del borde de área de protección y el valor del borde de área de cobertura ({tipo}: entre {minimo} y {maximo} dBuV/m).',
                                 line=dict(color=colors['Bet'])))
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['minus'], fill='tozeroy',
                                 name=f'Los valores de campo eléctrico diario son inferiores al valor del borde de área de protección ({tipo}: <{minimo} dBuV/m).',
                                 line=dict(color=colors['Minus'])))
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['valor'], fill='tozeroy',
                                 name='No se dispone de mediciones del sistema SACER.',
                                 line=dict(color=colors['Valor'])))

        # Use the autorizations_selected flag to determine whether to plot 'autorizaciones' data
        if autorizations_selected:
            fig.add_trace(
                go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['aut'], fill='tozeroy',
                           name='Dispone de autorización para suspensión de emisiones y/o operación con baja potencia.',
                           line=dict(color=colors['Autorizaciones'])))

        # Setting plot layout
        tick_labels = df_filtered['Tiempo'].unique().tolist()
        tick_positions = list(range(len(tick_labels)))  # Convert range to list

        fig.update_layout(
            title=f'Ciudad: {ciudad}, Estación: {nombre}, Frecuencia: {frequency} Hz',
            xaxis=dict(
                title='Tiempo',
                type='category',
                tickangle=-45,
                tickmode='auto',
                nticks=31,
                ticktext=tick_labels,
                tickvals=tick_positions
            ),
            yaxis=dict(
                title='Nivel de Intensidad de Campo Eléctrico (dBµV/m)',
                range=[0, 120]
            ),
            margin=dict(l=100, r=100, t=100, b=100),
            hovermode='closest',
            legend=dict(
                x=0.5,
                y=-0.3,
                traceorder='normal',
                font=dict(
                    size=12,
                ),
                orientation='h',
                xanchor='center',
                yanchor='top'
            )
        )

        # Adding annotations for initial and final dates of the authorization
        if autorizations_selected:
            for mark_time in df_filtered.Fecha_inicio.unique():
                if mark_time and mark_time != 0:  # Ensure this is the correct condition
                    try:
                        mark_index = tick_labels.index(mark_time)
                        fig.add_annotation(
                            x=mark_index, y=0,
                            text=f'Inicio: {mark_time}',
                            showarrow=True,
                            arrowhead=1,
                            ax=0, ay=-60,  # Arrow direction
                            bgcolor="white",  # Background color of the text box
                            bordercolor="black",  # Border color of the text box
                            font=dict(color="black")  # Text font color
                        )
                    except ValueError:
                        pass  # If mark_time is not in tick_labels, skip

            for mark_time in df_filtered.Fecha_fin.unique():
                if mark_time and mark_time != 0:  # Ensure this is the correct condition
                    try:
                        mark_index = tick_labels.index(mark_time)
                        fig.add_annotation(
                            x=mark_index, y=0,
                            text=f'Fin: {mark_time}',
                            showarrow=True,
                            arrowhead=1,
                            ax=0, ay=-30,  # Arrow direction
                            bgcolor="white",  # Background color of the text box
                            bordercolor="black",  # Border color of the text box
                            font=dict(color="black")  # Text font color
                        )
                    except ValueError:
                        pass  # If mark_time is not in tick_labels, skip

        plots.append(dcc.Graph(figure=fig))

    # Return a Div containing all the plots
    return html.Div(plots)


def update_station_plot_tv(selected_frequencies: list, stored_data: list, autorizations_selected: bool,
                           ciudad: str) -> dcc.Graph:
    """
    Update station plot for TV based on selected frequencies, stored data, and authorization status.

    This function generates a series of interactive plots for TV frequencies, showing the daily electric field levels
    and highlighting areas based on different criteria such as authorization status and level thresholds.

    Args:
        selected_frequencies (list): List of selected frequencies to plot.
        stored_data (list): List of dictionaries containing stored data for plotting.
        autorizations_selected (bool): Flag indicating whether to include data related to authorization status in plots.
        ciudad (str): Name of the city for which the plot is being generated.

    Returns:
        dcc.Graph: A Dash core component Graph that contains the interactive plot.
    """
    if not selected_frequencies or not stored_data:
        return no_update

    # Convert stored_data to DataFrame
    df = pd.DataFrame.from_records(stored_data)
    df['Tiempo'] = pd.to_datetime(df['Tiempo'], errors='coerce')
    df = df.sort_values(by='Tiempo', ascending=True)
    df['Tiempo'] = df['Tiempo'].dt.strftime('%Y-%m-%d')
    # Container for the plots
    plots = []

    # Create a plot for each selected frequency
    for frequency in selected_frequencies:
        df_filtered = df[df['Frecuencia (Hz)'] == frequency]
        df_filtered = df_filtered.rename(
            columns={'Frecuencia (Hz)': 'freq', 'Estación': 'est', 'Canal (Número)': 'canal', 'Analógico/Digital': 'ad',
                     'Level (dBµV/m)': 'level', 'Inicio Autorización': 'Fecha_inicio', 'Fin Autorización': 'Fecha_fin'})
        df_filtered['Fecha_inicio'] = df_filtered['Fecha_inicio'].fillna(0)
        df_filtered['Fecha_fin'] = df_filtered['Fecha_fin'].fillna(0)
        nombre = df_filtered['est'].iloc[0]
        andig = df_filtered['ad'].iloc[0]
        if andig == '-':
            andig = 0
        else:
            andig = 1

        def minus(row):
            try:
                level = float(row['level'])  # Convert level to float
            except ValueError:
                return 0  # Return 0 if conversion fails

            if level > 0:
                if (54000000 <= row['freq'] <= 88000000 and andig == 0 and level < 47) or \
                        (174000000 <= row['freq'] <= 216000000 and andig == 0 and level < 56) or \
                        (470000000 <= row['freq'] <= 880000000 and andig == 0 and level < 64) or \
                        (andig == 1 and level < 30):
                    return level
            return 0

        def bet(row):
            try:
                level = float(row['level'])  # Convert level to float
            except ValueError:
                return 0  # Return 0 if conversion fails

            if level > 0:
                if (54000000 <= row['freq'] <= 88000000 and andig == 0 and 47 <= level < 68) or \
                        (174000000 <= row['freq'] <= 216000000 and andig == 0 and 56 <= level < 71) or \
                        (470000000 <= row['freq'] <= 880000000 and andig == 0 and 64 <= level < 74) or \
                        (andig == 1 and 30 <= level < 51):
                    return level
            return 0

        def plus(row):
            try:
                level = float(row['level'])  # Convert level to float
            except ValueError:
                return 0  # Return 0 if conversion fails

            if level > 0:
                if (54000000 <= row['freq'] <= 88000000 and andig == 0 and level >= 68) or \
                        (174000000 <= row['freq'] <= 216000000 and andig == 0 and level >= 71) or \
                        (470000000 <= row['freq'] <= 880000000 and andig == 0 and level >= 74) or \
                        (andig == 1 and level >= 51):
                    return level
            return 0

        def valor(row):
            """function to return a specific value if the value in every row of the column 'level' meet the
            condition"""
            if row['level'] == 0:
                return 120
            return 0

        def aut(row):
            """function to return a specific value if the value in every row of the column 'level' meet the
            condition"""
            if row['Fecha_fin'] != 0 and row['level'] == 0:
                return 0
            elif row['Fecha_fin'] != 0 and row['level'] != 0:
                return row['level']
            return 0

        """create a new column in the df_filtered frame for every definition (minus, bet, plus, valor, aut)"""
        df_filtered['minus'] = df_filtered.apply(lambda row: minus(row), axis=1)
        df_filtered['bet'] = df_filtered.apply(lambda row: bet(row), axis=1)
        df_filtered['plus'] = df_filtered.apply(lambda row: plus(row), axis=1)
        df_filtered['valor'] = df_filtered.apply(lambda row: valor(row), axis=1)
        df_filtered['aut'] = df_filtered.apply(lambda row: aut(row), axis=1)
        # Creating the plot
        fig = go.Figure()

        colors = {
            'Plus': '#7fc97f',  # Example color, similar to 'Accent'
            'Bet': '#FFD700',  # Example color, similar to 'Set3_r'
            'Minus': '#ff9999',  # Example color, similar to 'Pastel1'
            'Valor': '#beaed4',  # Example color, similar to 'Paired'
            'Autorizaciones': '#386cb0'  # Example color, similar to 'Set2_r'
        }

        if df_filtered['freq'].iloc[0] >= 54000000 and df_filtered['freq'].iloc[0] <= 88000000 and andig == 0:
            Plus = f'Los valores de campo eléctrico diario superan el límite del área de cobertura primaria (Frecuencia {frequency} Hz: >= 68 dBuV/m).'
            Bet = f'Los valores de campo eléctrico diario superan el límite del área de cobertura secundario pero son inferiores al límite del área de cobertura principal establecido (Frecuencia {frequency} Hz: entre 47 y 68 dBuV/m).'
            Minus = f'Los valores de campo eléctrico diario son inferiores al límite de área de cobertura secundario. (Frecuencia {frequency} Hz: < 47 dBuV/m).'
        elif df_filtered['freq'].iloc[0] >= 174000000 and df_filtered['freq'].iloc[0] <= 216000000 and andig == 0:
            Plus = f'Los valores de campo eléctrico diario superan el límite del área de cobertura primaria (Frecuencia {frequency} Hz: >= 71 dBuV/m).'
            Bet = f'Los valores de campo eléctrico diario superan el límite del área de cobertura secundario pero son inferiores al límite del área de cobertura principal establecido (Frecuencia {frequency} Hz: entre 56 y 71 dBuV/m).'
            Minus = f'Los valores de campo eléctrico diario son inferiores al límite de área de cobertura secundario. (Frecuencia {frequency} Hz: < 56 dBuV/m).'
        elif df_filtered['freq'].iloc[0] >= 470000000 and df_filtered['freq'].iloc[0] <= 880000000 and andig == 0:
            Plus = f'Los valores de campo eléctrico diario superan el límite del área de cobertura primaria (Frecuencia {frequency} Hz: >= 74 dBuV/m).'
            Bet = f'Los valores de campo eléctrico diario superan el límite del área de cobertura secundario pero son inferiores al límite del área de cobertura principal establecido (Frecuencia {frequency} Hz: entre 64 y 74 dBuV/m).'
            Minus = f'Los valores de campo eléctrico diario son inferiores al límite de área de cobertura secundario. (Frecuencia {frequency} Hz: < 64 dBuV/m).'
        elif andig == 1:
            Plus = f'Los valores de campo eléctrico diario superan el límite del área de cobertura primaria (Frecuencia {frequency} Hz: >= 51 dBuV/m).'
            Bet = f'Los valores de campo eléctrico diario superan el límite del área de cobertura secundario pero son inferiores al límite del área de cobertura principal establecido (Frecuencia {frequency} Hz: entre 30 y 51 dBuV/m).'
            Minus = f'Los valores de campo eléctrico diario son inferiores al límite de área de cobertura secundario. (Frecuencia {frequency} Hz: < 30 dBuV/m).'

        Valor = 'No se dispone de mediciones del sistema SACER.'
        Autorizaciones = 'Dispone de autorización para suspensión de emisiones y/o operación con baja potencia.'

        # Adding area plots with custom colors
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['plus'], fill='tozeroy', name=Plus,
                                 line=dict(color=colors['Plus'])))
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['bet'], fill='tozeroy', name=Bet,
                                 line=dict(color=colors['Bet'])))
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['minus'], fill='tozeroy', name=Minus,
                                 line=dict(color=colors['Minus'])))
        fig.add_trace(go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['valor'], fill='tozeroy', name=Valor,
                                 line=dict(color=colors['Valor'])))

        # Use the autorizations_selected flag to determine whether to plot 'autorizaciones' data
        if autorizations_selected:
            fig.add_trace(
                go.Scatter(x=df_filtered['Tiempo'], y=df_filtered['aut'], fill='tozeroy', name=Autorizaciones,
                           line=dict(color=colors['Autorizaciones'])))

        # Setting plot layout
        tick_labels = df_filtered['Tiempo'].unique().tolist()
        tick_positions = list(range(len(tick_labels)))  # Convert range to list

        fig.update_layout(
            title=f'Ciudad: {ciudad}, Estación: {nombre}, Frecuencia: {frequency} Hz',
            xaxis=dict(
                title='Tiempo',
                type='category',
                tickangle=-45,
                tickmode='auto',
                nticks=31,
                ticktext=tick_labels,
                tickvals=tick_positions
            ),
            yaxis=dict(
                title='Nivel de Intensidad de Campo Eléctrico (dBµV/m)',
                range=[0, 120]
            ),
            margin=dict(l=100, r=100, t=100, b=100),
            hovermode='closest',
            legend=dict(
                x=0.5,
                y=-0.3,
                traceorder='normal',
                font=dict(
                    size=12,
                ),
                orientation='h',
                xanchor='center',
                yanchor='top'
            )
        )

        # Adding annotations for initial and final dates of the authorization
        if autorizations_selected:
            for mark_time in df_filtered.Fecha_inicio.unique():
                if mark_time and mark_time != 0:  # Ensure this is the correct condition
                    try:
                        mark_index = tick_labels.index(mark_time)
                        fig.add_annotation(
                            x=mark_index, y=0,
                            text=f'Inicio: {mark_time}',
                            showarrow=True,
                            arrowhead=1,
                            ax=0, ay=-60,  # Arrow direction
                            bgcolor="white",  # Background color of the text box
                            bordercolor="black",  # Border color of the text box
                            font=dict(color="black")  # Text font color
                        )
                    except ValueError:
                        pass  # If mark_time is not in tick_labels, skip

            for mark_time in df_filtered.Fecha_fin.unique():
                if mark_time and mark_time != 0:  # Ensure this is the correct condition
                    try:
                        mark_index = tick_labels.index(mark_time)
                        fig.add_annotation(
                            x=mark_index, y=0,
                            text=f'Fin: {mark_time}',
                            showarrow=True,
                            arrowhead=1,
                            ax=0, ay=-30,  # Arrow direction
                            bgcolor="white",  # Background color of the text box
                            bordercolor="black",  # Border color of the text box
                            font=dict(color="black")  # Text font color
                        )
                    except ValueError:
                        pass  # If mark_time is not in tick_labels, skip

        plots.append(dcc.Graph(figure=fig))

    # Return a Div containing all the plots
    return html.Div(plots)
