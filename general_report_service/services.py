import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
from django.conf import settings
from pandas import DataFrame

from .utils import translate_month, read_csv_file
from .api.api_client import get_options_from_index_service_api

CITIES1 = settings.CITIES1


def get_options(self) -> dict:
    """
    Retrieve options from the index service API.

    Returns:
        dict: Options obtained from the API.
    """
    return get_options_from_index_service_api()


def customize_data(selected_options: dict) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Customize data based on selected city and date range. This function performs several operations:
    - Retrieves data based on city and date range.
    - Cleans and processes the data for FM, TV, and AM broadcasting.
    - Concatenates data from different sources.
    - Filters and simplifies data for the specified time period and authority.

    Args:
        selected_options (dict): Dictionary containing 'city', 'start_date', and 'end_date'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing six dataframes:
            - Three 'original' dataframes with cleaned data for FM, TV, and AM broadcasting.
            - Three 'clean' dataframes with further processed data.
    """
    ciudad = selected_options['city']
    fecha_inicio = selected_options['start_date']
    fecha_fin = selected_options['end_date']

    ciu, autori, sheet_name1, sheet_name2, *rest = CITIES1.get(ciudad, (None, None, None, None, None))
    sheet_name3 = rest[0] if rest else None  # Assign sheet_name3 only if rest is not empty

    def convert(date_time: str) -> datetime.datetime:
        """
        Convert date string to datetime object.

        Args:
            date_time (str): Date string.

        Returns:
            datetime.datetime: Corresponding datetime object.
        """
        format_str = '%Y-%m-%d %H:%M:%S' if ' ' in date_time else '%Y-%m-%d'
        datetime_obj = datetime.datetime.strptime(date_time, format_str)
        return datetime_obj

    Year1, Year2 = convert(fecha_inicio).year, convert(fecha_fin).year

    month_year = generate_month_year_vector(Year1, Year2)

    df_data1, df_data2, df_data3 = read_data_files(selected_options, ciu, month_year, sheet_name1, sheet_name2,
                                                   sheet_name3)

    dfau = pd.concat([read_and_process_aut(settings.FILE_AUT_SUS, settings.COLUMNS_AUT, 'S'),
                      read_and_process_aut(settings.FILE_AUT_BP, settings.COLUMNS_AUTBP, 'BP')],
                     ignore_index=True)

    df_origin1 = pd.DataFrame(df_data1, columns=settings.COLUMNS_FM)
    df_origin1['Tiempo'] = pd.to_datetime(df_origin1['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
    df_clean1 = clean_data(fecha_inicio, fecha_fin, df_origin1, sheet_name1)
    df_original1 = simplify_fm_broadcasting(df_clean1, dfau, autori)
    df_clean1 = df_clean1.drop(
        columns=['Offset (Hz)', 'FM (Hz)', 'Bandwidth (Hz)', 'Estación', 'Potencia', 'BW Asignado'])

    df_origin2 = pd.DataFrame(df_data2, columns=settings.COLUMNS_TV)
    df_origin2['Tiempo'] = pd.to_datetime(df_origin2['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
    df_clean2 = clean_data(fecha_inicio, fecha_fin, df_origin2, sheet_name2)
    df_original2 = simplify_tv_broadcasting(df_clean2, dfau, autori)
    df_clean2 = df_clean2.drop(
        columns=['Offset (Hz)', 'AM (%)', 'Bandwidth (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital'])

    if df_data3 is not None:
        df_origin3 = pd.DataFrame(df_data3, columns=settings.COLUMNS_AM)
        df_origin3['Tiempo'] = pd.to_datetime(df_origin3['Tiempo'], format="%d/%m/%Y %H:%M:%S.%f")
        df_clean3 = clean_data(fecha_inicio, fecha_fin, df_origin3, sheet_name3)
        df_original3 = simplify_am_broadcasting(df_clean3, dfau, autori)
        df_clean3 = df_clean3.drop(
            columns=['Offset (Hz)', 'AM (%)', 'Bandwidth (Hz)', 'Estación'])
    else:
        df_clean3 = pd.DataFrame()
        df_original3 = pd.DataFrame()
    return df_original1, df_original2, df_original3, df_clean1, df_clean2, df_clean3


def generate_month_year_vector(year1: int, year2: int) -> list[str]:
    """
    Generate a vector of month-year strings between the given years.

    Args:
        year1 (int): Start year.
        year2 (int): End year.

    Returns:
        List[str]: List of month-year strings.
    """
    vector = []
    for year in range(year1, year2 + 1):
        meses = [f"{translate_month(datetime.date(year, month, 1).strftime('%B'))}_{year}" for month in
                 range(1, 13)]
        vector.append(meses)
    return [num for elem in vector for num in elem]


def read_data_files(selected_options: dict, ciu: str, month_year: list[str], sheet_name1: str,
                    sheet_name2: str, sheet_name3: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read data files based on selected options and return data within the specified date range.

    Args:
        selected_options (dict): Selected options.
        ciu (str): City.
        month_year (List[str]): List of month-year strings.
        sheet_name1 (str): Sheet name for data set 1.
        sheet_name2 (str): Sheet name for data set 2.
        sheet_name3 (str): Sheet name for data set 3.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Data arrays within the selected date range.
    """
    # Convert start and end dates from string to datetime objects
    start_date = datetime.datetime.strptime(selected_options['start_date'], '%Y-%m-%d')
    end_date = datetime.datetime.strptime(selected_options['end_date'], '%Y-%m-%d')
    # Calculate the start and end indices for the month_year based on selected_options
    start_idx = month_year.index(f"{translate_month(start_date.strftime('%B'))}_{start_date.year}")
    end_idx = month_year.index(f"{translate_month(end_date.strftime('%B'))}_{end_date.year}")

    # Initialize empty lists to store data
    df_data1, df_data2, df_data3 = [], [], []

    # Iterate over the relevant months and read data
    for mes in month_year[start_idx:end_idx + 1]:
        df_data1.append(read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/FM_{ciu}_{mes}.csv', settings.COLUMNS_FM))
        df_data2.append(read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/TV_{ciu}_{mes}.csv', settings.COLUMNS_TV))

        if sheet_name3:
            df_data3.append(
                read_csv_file(f'{settings.SERVER_ROUTE}/{ciu}/AM_{ciu}_{mes}.csv', settings.COLUMNS_AM))

    # Concatenate the data and handle None for df_d3
    df_data1 = dd.concat(df_data1).compute() if df_data1 else dd.from_array(np.array([]))
    df_data2 = dd.concat(df_data2).compute() if df_data2 else dd.from_array(np.array([]))
    df_data3 = dd.concat(df_data3).compute() if df_data3 else None

    return df_data1, df_data2, df_data3


def read_and_process_aut(file_name: str, column_names: list[str], kind: str) -> pd.DataFrame:
    """
    Read and process authorization data.

    Args:
        file_name (str): Name of the authorization file.
        column_names (List[str]): List of column names.
        kind (str): Type of authorization.

    Returns:
        pd.DataFrame: Processed authorization DataFrame.
    """
    df = pd.read_excel(f'{settings.SERVER_ROUTE}/{file_name}', skiprows=1, usecols=column_names)
    df = df.fillna('-')
    df = df.rename(columns={
        'FECHA INGRESO': 'Fecha_ingreso',
        'FREC / CANAL': 'freq1',
        'CIUDAD PRINCIPAL COBERTURA': 'ciu',
        'No. OFICIO ARCOTEL': 'Oficio',
        'NOMBRE ESTACIÓN': 'est',
        f'FECHA INICIO {"SUSPENSION" if kind == "S" else "BAJA POTENCIA"}': 'Fecha_inicio',
        'FECHA OFICIO': 'Fecha_oficio',
        'DIAS': 'Plazo'
    })
    df['Tipo'] = pd.Series([kind for _ in range(len(df.index))])
    df = df[df.Oficio != '-']
    df = df[df.Fecha_inicio != '-']
    df['Fecha_ingreso'] = df['Fecha_ingreso'].replace({'-': ''})
    df['Fecha_ingreso'] = pd.to_datetime(df['Fecha_ingreso'])
    df['Fecha_oficio'] = df['Fecha_oficio'].replace({'-': ''})
    df['Fecha_oficio'] = pd.to_datetime(df['Fecha_oficio'])
    df['Fecha_inicio'] = df['Fecha_inicio'].replace({'-': ''})
    df['Fecha_inicio'] = pd.to_datetime(df['Fecha_inicio'])
    df['Fecha_fin'] = df['Fecha_inicio'] + pd.to_timedelta(df['Plazo'] - 1, unit='d')
    df['freq1'] = df['freq1'].replace('-', np.nan)
    df['freq1'] = pd.to_numeric(df['freq1'])

    def freq(row: pd.Series) -> float:
        """
        Modify the frequency values in the 'freq1' column to present all in Hz, except for TV channel numbers.
        - If the frequency is between 570 and 1590, it's assumed to be in kHz and is converted to Hz by multiplying by 1000.
        - If the frequency is between 88 and 108, it's assumed to be in MHz and is converted to Hz by multiplying by 1000000.
        - Frequencies outside these ranges are assumed to be TV channel numbers or already in Hz and are returned as is.

        Args:
            row (pd.Series): A row of a pandas DataFrame.

        Returns:
            float: The frequency value converted to Hz if applicable, otherwise the original value.
        """
        if row['freq1'] >= 570 and row['freq1'] <= 1590:
            return row['freq1'] * 1000
        elif row['freq1'] >= 88 and row['freq1'] <= 108:
            return row['freq1'] * 1000000
        else:
            return row['freq1']

    # Create a new column in the df dataframe using the last function def freq(row)
    df['freq'] = df.apply(lambda row: freq(row), axis=1)
    df['freq'] = df['freq'].astype('float64')
    df = df.drop(columns=['freq1'])

    return df


def clean_data(start_date: str, end_date: str, df_primary: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Clean and preprocess data based on specified date range.

    Args:
        start_date (str): Start date.
        end_date (str): End date.
        df_primary (pd.DataFrame): Primary DataFrame.
        sheet_name (str): Sheet name.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_tx = pd.read_excel(f'{settings.SERVER_ROUTE}/{settings.FILE_ESTACIONES}', sheet_name=sheet_name).fillna('-')
    df_tx['Frecuencia (Hz)'] = df_tx['Frecuencia (Hz)'].astype('float64')

    # Convert date strings to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Ensure 'Tiempo' in df_primary is a datetime
    df_primary['Tiempo'] = pd.to_datetime(df_primary['Tiempo'])

    # Create a date range for the entire period
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Find the missing dates by comparing the all_dates with the unique dates in df_primary
    existing_dates = df_primary['Tiempo'].dt.normalize().unique()
    missing_dates = all_dates.difference(existing_dates)

    # If there are missing dates, create a DataFrame to represent them
    if not missing_dates.empty:
        df_missing_dates = pd.DataFrame(
            [(t, f) for t in missing_dates
             for f in df_tx['Frecuencia (Hz)'].tolist()],
            columns=('Tiempo', 'Frecuencia (Hz)')
        )
        df_missing_dates['Frecuencia (Hz)'] = df_missing_dates['Frecuencia (Hz)'].astype('float64')

        # Merge the missing dates DataFrame with df_primary
        df_primary = pd.concat([df_primary, df_missing_dates], ignore_index=True)

    # Merge the primary DataFrame with the transmission info DataFrame (df_tx)
    df_complete = df_primary.merge(df_tx, how='left', on='Frecuencia (Hz)')

    # Filter the data to the desired date range and fill missing values
    df_complete = df_complete[(df_complete['Tiempo'] >= start_date) & (df_complete['Tiempo'] <= end_date)].fillna(0)

    return df_complete


def simplify_fm_broadcasting(df: pd.DataFrame, dfau: pd.DataFrame, autori: str) -> DataFrame:
    """
    Simplify FM broadcasting data.

    Args:
        df9 (pd.DataFrame): FM broadcasting DataFrame.
        dfau1 (pd.DataFrame): Authorization DataFrame.
        autori (str): Authorization type.

    Returns:
        pd.DataFrame: Simplified DataFrame.
    """
    df_authorization = process_authorization_am_fm_df(dfau, 87700000, 108100000, autori)
    df_merge_data_aut = merge_authorization_with_data(df_authorization, df, ['Tiempo', 'Frecuencia (Hz)'])
    df_data_aut = df_merge_data_aut.groupby(['Frecuencia (Hz)', 'Estación', pd.Grouper(key='Tiempo', freq='D')]).agg(
        {'Level (dBµV/m)': 'max', 'Bandwidth (Hz)': np.average, 'Fecha_fin': 'max'})
    df_data_aut = df_data_aut.reset_index().sort_values(by='Tiempo', ascending=True)
    new_order = ['Tiempo', 'Frecuencia (Hz)', 'Estación', 'Level (dBµV/m)', 'Bandwidth (Hz)', 'Fecha_fin']
    df_data_aut = df_data_aut[new_order]
    return df_data_aut


def simplify_tv_broadcasting(df: pd.DataFrame, dfau: pd.DataFrame, autori: str) -> pd.DataFrame:
    """
    Simplify TV broadcasting data.

    Args:
        df10 (pd.DataFrame): TV broadcasting DataFrame.
        dfau1 (pd.DataFrame): Authorization DataFrame.
        autori (str): Authorization type.

    Returns:
        pd.DataFrame: Simplified DataFrame.
    """
    df_authorization = process_authorization_tv_df(dfau, 2, 51, autori)
    df_merge_data_aut = merge_authorization_with_data(df_authorization, df, ['Tiempo', 'Canal (Número)'])
    df_data_aut = df_merge_data_aut.groupby(
        ['Frecuencia (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital', pd.Grouper(key='Tiempo', freq='D')]).agg(
        {'Level (dBµV/m)': 'max', 'Fecha_fin': 'max'})
    df_data_aut = df_data_aut.reset_index().sort_values(by='Tiempo', ascending=True)
    new_order = ['Tiempo', 'Frecuencia (Hz)', 'Estación', 'Canal (Número)', 'Analógico/Digital', 'Level (dBµV/m)',
                 'Fecha_fin']
    df_data_aut = df_data_aut[new_order]
    return df_data_aut


def simplify_am_broadcasting(df: pd.DataFrame, dfau: pd.DataFrame, autori: str) -> pd.DataFrame:
    """
    Simplify AM broadcasting data.

    Args:
        df17 (pd.DataFrame): AM broadcasting DataFrame.
        dfau1 (pd.DataFrame): Authorization DataFrame.
        autori (str): Authorization type.

    Returns:
        pd.DataFrame: Simplified DataFrame.
    """
    df_authorization = process_authorization_am_fm_df(dfau, 570000, 1590000, autori)
    df_merge_data_aut = merge_authorization_with_data(df_authorization, df, ['Tiempo', 'Frecuencia (Hz)'])
    df_data_aut = df_merge_data_aut.groupby(['Frecuencia (Hz)', 'Estación', pd.Grouper(key='Tiempo', freq='D')]).agg(
        {'Level (dBµV/m)': 'max', 'Bandwidth (Hz)': np.average, 'Fecha_fin': 'max'})
    df_data_aut = df_data_aut.reset_index().sort_values(by='Tiempo', ascending=True)
    new_order = ['Tiempo', 'Frecuencia (Hz)', 'Estación', 'Level (dBµV/m)', 'Bandwidth (Hz)', 'Fecha_fin']
    df_data_aut = df_data_aut[new_order]
    return df_data_aut


def process_authorization_am_fm_df(dfau: pd.DataFrame, freq_range_start: int, freq_range_end: int,
                                   city: str) -> pd.DataFrame:
    """
    Process authorization data for AM/FM broadcasting.

    Args:
        dfau (pd.DataFrame): Authorization DataFrame.
        freq_range_start (int): Start of frequency range.
        freq_range_end (int): End of frequency range.
        city (str): City name.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    dfau_filtered = dfau[(dfau.freq >= freq_range_start) & (dfau.freq <= freq_range_end)]
    dfau_filtered = dfau_filtered.rename(columns={'freq': 'Frecuencia (Hz)', 'Fecha_inicio': 'Tiempo'})
    dfau_filtered = dfau_filtered.loc[dfau_filtered['ciu'] == city]
    dfau_filtered = dfau_filtered.drop(columns=['est'])
    dfau_filtered['Frecuencia (Hz)'] = dfau_filtered['Frecuencia (Hz)'].astype('float64')

    result_df = []
    for index, row in dfau_filtered.iterrows():
        for t in pd.date_range(start=row['Tiempo'], end=row['Fecha_fin']):
            result_df.append(
                (row['Frecuencia (Hz)'], row['Tipo'], row['Plazo'], t, row['Oficio'], row['Fecha_oficio'],
                 row['Fecha_fin'])
            )

    return pd.DataFrame(result_df, columns=(
        'Frecuencia (Hz)', 'Tipo', 'Plazo', 'Tiempo', 'Oficio', 'Fecha_oficio', 'Fecha_fin'))


def process_authorization_tv_df(dfau: pd.DataFrame, freq_range_start: int, freq_range_end: int,
                                city: str) -> pd.DataFrame:
    """
    Process authorization data for TV broadcasting.

    Args:
        dfau (pd.DataFrame): Authorization DataFrame.
        freq_range_start (int): Start of frequency range.
        freq_range_end (int): End of frequency range.
        city (str): City name.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    dfau_filtered = dfau[(dfau.freq >= freq_range_start) & (dfau.freq <= freq_range_end)]
    dfau_filtered = dfau_filtered.rename(columns={'freq': 'Canal (Número)', 'Fecha_inicio': 'Tiempo'})
    dfau_filtered = dfau_filtered.loc[dfau_filtered['ciu'] == city]
    dfau_filtered = dfau_filtered.drop(columns=['est'])
    dfau_filtered['Canal (Número)'] = dfau_filtered['Canal (Número)'].astype('float64')

    result_df = []
    for index, row in dfau_filtered.iterrows():
        for t in pd.date_range(start=row['Tiempo'], end=row['Fecha_fin']):
            result_df.append(
                (row['Canal (Número)'], row['Tipo'], row['Plazo'], t, row['Oficio'], row['Fecha_oficio'],
                 row['Fecha_fin'])
            )

    return pd.DataFrame(result_df, columns=(
        'Canal (Número)', 'Tipo', 'Plazo', 'Tiempo', 'Oficio', 'Fecha_oficio', 'Fecha_fin'))


def merge_authorization_with_data(df_authorization: pd.DataFrame, df_data: pd.DataFrame,
                                  merge_columns: list[str]) -> pd.DataFrame:
    """
    Merge authorization data with broadcasting data.

    Args:
        df_authorization (pd.DataFrame): Authorization DataFrame.
        df_data (pd.DataFrame): Broadcasting DataFrame.
        merge_columns (List[str]): List of columns to merge on.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    df_authorization = df_authorization.rename(columns={'Fecha_inicio': 'Tiempo'})

    result_df = df_authorization.merge(df_data, how='right', on=merge_columns)
    return result_df.fillna('-')


def create_pivot_table(df: pd.DataFrame, index: list, values: list, columns: list,
                       aggfunc: dict) -> pd.DataFrame:
    """
    Create a pivot table with specified parameters.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - index: list
        List of index columns in the form [pd.Grouper(key='ColumnName')].
    - values: list
        List of values to be aggregated.
    - columns: list
        List of columns to be used for pivoting.
    - aggfunc: dict
        Dictionary specifying the aggregation functions for each value.

    Returns:
    - DataFrame
        The resulting pivot table.
    """
    # Make the pivot table with the data structured in the way we want to show in the report
    pivot_table = pd.pivot_table(df, index=index, values=values, columns=columns, aggfunc=aggfunc).round(2)

    # Rename column
    pivot_table = pivot_table.rename(columns={'Fecha_fin': 'Fin de Autorización'})

    # Transpose the DataFrame
    pivot_table = pivot_table.T

    # Replace 0 with '-'
    pivot_table = pivot_table.replace(0, '-')

    # Reset the index (unstack)
    pivot_table = pivot_table.reset_index()

    # Sort the DataFrame
    sorter = ['Level (dBµV/m)', 'Bandwidth (Hz)', 'Fin de Autorización']
    pivot_table.level_0 = pivot_table.level_0.astype("category")
    pivot_table.level_0 = pivot_table.level_0.cat.set_categories(sorter)
    pivot_table = pivot_table.sort_values(['level_0', 'Frecuencia (Hz)'])

    # Rename columns
    final_result = pivot_table.rename(columns={'level_0': 'Param'})

    return final_result
