import datetime
import pandas as pd
import psycopg
from django.conf import settings
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from .utils import convert_start_date, convert_end_date, process_warnings_data

pd.set_option('future.no_silent_downcasting', True)


def get_db_connection():
    """
    Get a connection to the PostgreSQL database.

    Returns:
        psycopg.Connection: A connection to the PostgreSQL database.
    """
    try:
        conn = psycopg.connect(
            dbname=settings.DBNAME,
            user=settings.USER,
            password=settings.PASSWORD,
            host=settings.HOST,
            port=settings.PORT
        )
        return conn
    except Exception as e:
        # Log the error and raise a custom exception
        raise Exception("Error connecting to the database: {}".format(e))


@cache_page(60 * 10)
def fm_fetch_data_from_db(request):
    """
    Fetch FM data from the database based on the selected city and date range.

    Args:
        request (django.http.HttpRequest): The HTTP request object.

    Returns:
        pandas.DataFrame: A DataFrame containing the FM data.
    """
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    city = request.GET.get('city')
    last_query_time = request.GET.get('last_query_time')

    # Convert start_date and end_date to datetime objects
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    # Convert last_query_time to datetime object if it exists
    if last_query_time:
        last_query_time = datetime.datetime.strptime(last_query_time, '%Y-%m-%d %H:%M:%S')

    # Check if the data is already in the cache
    cache_key = f"fm_data_{city}_{start_date}_{end_date}_{last_query_time}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    try:
        conn = get_db_connection()
        query = """
            SELECT *
            FROM radio_fm_processed_info
            WHERE city = %s AND tiempo >= %s AND tiempo < %s + INTERVAL '1 DAY'
        """
        if last_query_time is not None:
            query += " AND tiempo > %s"
            params = [city, start_date, end_date, last_query_time]
        else:
            params = [city, start_date, end_date]
        df = pd.read_sql(query, conn, params=params)
        conn.close()

        # Eliminar filas con valores nulos en 'level_dbuv_m'
        df = df.dropna(subset=['level_dbuv_m'])

        # Eliminar filas duplicadas
        df = df.drop_duplicates()

        # # Analizar el DataFrame
        # df_analysis = analyze_dataframe(df)
        #
        # # Imprimir o loggear el análisis
        # print("DataFrame después de la limpieza:")
        # print(df_analysis)

        # Cache the data for 10 minutes
        cache.set(cache_key, df, 60 * 10)

        return df
    except Exception as e:
        # Log the error and raise a custom exception
        raise Exception("Error fetching FM data from the database: {}".format(e))


@cache_page(60 * 10)
def tv_fetch_data_from_db(request):
    """
    Fetch TV data from the database based on the selected city and date range.

    Args:
        request (django.http.HttpRequest): The HTTP request object.

    Returns:
        pandas.DataFrame: A DataFrame containing the TV data.
    """
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    city = request.GET.get('city')
    last_query_time = request.GET.get('last_query_time')

    # Convert start_date and end_date to datetime objects
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    # Convert last_query_time to datetime object if it exists
    if last_query_time:
        last_query_time = datetime.datetime.strptime(last_query_time, '%Y-%m-%d %H:%M:%S')

    # Check if the data is already in the cache
    cache_key = f"tv_data_{city}_{start_date}_{end_date}_{last_query_time}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    try:
        conn = get_db_connection()
        query = """
            SELECT *
            FROM tv_processed_info
            WHERE city = %s AND tiempo >= %s AND tiempo < %s + INTERVAL '1 DAY'
        """
        if last_query_time is not None:
            query += " AND tiempo > %s"
            params = [city, start_date, end_date, last_query_time]
        else:
            params = [city, start_date, end_date]
        df = pd.read_sql(query, conn, params=params)
        conn.close()

        # Eliminar filas con valores nulos en 'level_dbuv_m'
        df = df.dropna(subset=['level_dbuv_m'])

        # Eliminar filas duplicadas
        df = df.drop_duplicates()

        # # Analizar el DataFrame
        # df_analysis = analyze_dataframe(df)
        #
        # # Imprimir o loggear el análisis
        # print("DataFrame después de la limpieza:")
        # print(df_analysis)

        # Cache the data for 10 minutes
        cache.set(cache_key, df, 60 * 10)

        return df
    except Exception as e:
        # Log the error and raise a custom exception
        raise Exception("Error fetching TV data from the database: {}".format(e))


@cache_page(60 * 10)
def am_fetch_data_from_db(request):
    """
    Fetch AM data from the database based on the selected city and date range.

    Args:
        request (django.http.HttpRequest): The HTTP request object.

    Returns:
        pandas.DataFrame: A DataFrame containing the FM data.
    """
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    city = request.GET.get('city')
    last_query_time = request.GET.get('last_query_time')

    # Convert start_date and end_date to datetime objects
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    # Convert last_query_time to datetime object if it exists
    if last_query_time:
        last_query_time = datetime.datetime.strptime(last_query_time, '%Y-%m-%d %H:%M:%S')

    # Check if the data is already in the cache
    cache_key = f"am_data_{city}_{start_date}_{end_date}_{last_query_time}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    try:
        conn = get_db_connection()
        query = """
            SELECT *
            FROM radio_am_processed_info
            WHERE city = %s AND tiempo >= %s AND tiempo < %s + INTERVAL '1 DAY'
        """
        if last_query_time is not None:
            query += " AND tiempo > %s"
            params = [city, start_date, end_date, last_query_time]
        else:
            params = [city, start_date, end_date]
        df = pd.read_sql(query, conn, params=params)
        conn.close()

        # Eliminar filas con valores nulos en 'level_dbuv_m'
        df = df.dropna(subset=['level_dbuv_m'])

        # Eliminar filas duplicadas
        df = df.drop_duplicates()

        # # Analizar el DataFrame
        # df_analysis = analyze_dataframe(df)
        #
        # # Imprimir o loggear el análisis
        # print("DataFrame después de la limpieza:")
        # print(df_analysis)

        # Cache the data for 10 minutes
        cache.set(cache_key, df, 60 * 10)

        return df
    except Exception as e:
        # Log the error and raise a custom exception
        raise Exception("Error fetching AM data from the database: {}".format(e))


def customize_data(request) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Customize data based on selected city and date range. This function performs several operations:
    - Retrieves data based on city and date range.
    - Cleans and processes the data for FM, TV, and AM broadcasting.
    - Concatenates data from different sources.
    - Filters and simplifies data for the specified time period and authority.

    Args:
        request (django.http.HttpRequest): The HTTP request object.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing six dataframes:
            - Three 'original' dataframes with cleaned data for FM, TV, and AM broadcasting.
            - Three 'clean' dataframes with further processed data.
    """
    ciudad = request.GET.get('city')
    fecha_inicio = convert_start_date(request.GET.get('start_date'))
    fecha_fin = convert_end_date(request.GET.get('end_date'))

    df_data1 = fm_fetch_data_from_db(request)

    if df_data1 is None:
        df_original1 = pd.DataFrame()
        df_clean1 = pd.DataFrame()
    else:
        df_original1 = df_data1.copy()
        df_clean1 = clean_data(fecha_inicio, fecha_fin, df_original1)
        df_original1 = df_clean1.sort_values(by='tiempo', ascending=False).fillna('-')
        df_original1 = df_original1.rename(
            columns={'tiempo': 'Tiempo', 'frecuencia_hz': 'Frecuencia (Hz)', 'level_dbuv_m': 'Level (dBµV/m)',
                     'bandwidth_hz': 'Bandwidth (Hz)', 'city': 'Ciudad', 'estacion': 'Estación', 'potencia': 'Potencia',
                     'bw_asignado': 'BW Asignado', 'fecha_ingreso': 'Fecha Ingreso', 'ingreso': 'Ingreso',
                     'oficio': 'Oficio', 'fecha_oficio': 'Fecha Oficio', 'plazo': 'Plazo',
                     'fecha_inicio': 'Inicio Autorización', 'fecha_fin': 'Fin Autorización', 'tipo': 'Tipo'})
        df_clean1 = df_clean1.rename(
            columns={'tiempo': 'Tiempo', 'frecuencia_hz': 'Frecuencia (Hz)', 'level_dbuv_m': 'Level (dBµV/m)',
                     'bandwidth_hz': 'Bandwidth (Hz)', 'city': 'Ciudad', 'estacion': 'Estación', 'potencia': 'Potencia',
                     'bw_asignado': 'BW Asignado', 'fecha_ingreso': 'Fecha Ingreso', 'ingreso': 'Ingreso',
                     'oficio': 'Oficio', 'fecha_oficio': 'Fecha Oficio', 'plazo': 'Plazo',
                     'fecha_inicio': 'Inicio Autorización', 'fecha_fin': 'Fin Autorización', 'tipo': 'Tipo'})
        df_clean1 = df_clean1.drop(
            columns=['Bandwidth (Hz)', 'Potencia', 'BW Asignado', 'Fecha Ingreso', 'Ingreso', 'Oficio', 'Fecha Oficio',
                     'Plazo', 'Inicio Autorización', 'Fin Autorización', 'Tipo'])
        df_clean1 = df_clean1.groupby(['Frecuencia (Hz)', pd.Grouper(key='Tiempo', freq='D')]).agg(
            {'Level (dBµV/m)': 'max', 'Estación': 'first'}).reset_index()

    df_data2 = tv_fetch_data_from_db(request)

    if df_data2 is None:
        df_original2 = pd.DataFrame()
        df_clean2 = pd.DataFrame()
    else:
        df_original2 = df_data2.copy()
        df_clean2 = clean_data(fecha_inicio, fecha_fin, df_original2)
        df_original2 = df_clean2.sort_values(by='tiempo', ascending=False).fillna('-')
        df_original2 = df_original2.rename(
            columns={'tiempo': 'Tiempo', 'frecuencia_hz': 'Frecuencia (Hz)', 'level_dbuv_m': 'Level (dBµV/m)',
                     'bandwidth_hz': 'Bandwidth (Hz)', 'city': 'Ciudad', 'estacion': 'Estación',
                     'canal_numero': 'Canal (Número)', 'analogico_digital': 'Analógico/Digital',
                     'fecha_ingreso': 'Fecha Ingreso', 'ingreso': 'Ingreso', 'oficio': 'Oficio',
                     'fecha_oficio': 'Fecha Oficio', 'plazo': 'Plazo', 'fecha_inicio': 'Inicio Autorización',
                     'fecha_fin': 'Fin Autorización', 'tipo': 'Tipo'})
        df_clean2 = df_clean2.rename(
            columns={'tiempo': 'Tiempo', 'frecuencia_hz': 'Frecuencia (Hz)', 'level_dbuv_m': 'Level (dBµV/m)',
                     'bandwidth_hz': 'Bandwidth (Hz)', 'city': 'Ciudad', 'estacion': 'Estación',
                     'canal_numero': 'Canal (Número)', 'analogico_digital': 'Analógico/Digital',
                     'fecha_ingreso': 'Fecha Ingreso', 'ingreso': 'Ingreso', 'oficio': 'Oficio',
                     'fecha_oficio': 'Fecha Oficio', 'plazo': 'Plazo', 'fecha_inicio': 'Inicio Autorización',
                     'fecha_fin': 'Fin Autorización', 'tipo': 'Tipo'})
        df_clean2 = df_clean2.drop(
            columns=['Bandwidth (Hz)', 'Canal (Número)', 'Analógico/Digital', 'Fecha Ingreso', 'Ingreso', 'Oficio',
                     'Fecha Oficio', 'Plazo', 'Inicio Autorización', 'Fin Autorización', 'Tipo'])
        df_clean2 = df_clean2.groupby(['Frecuencia (Hz)', pd.Grouper(key='Tiempo', freq='D')]).agg(
            {'Level (dBµV/m)': 'max', 'Estación': 'first'}).reset_index()

    df_data3 = am_fetch_data_from_db(request)

    if df_data3 is None:
        df_original3 = pd.DataFrame()
        df_clean3 = pd.DataFrame()
    else:
        df_original3 = df_data3.copy()
        df_clean3 = clean_data(fecha_inicio, fecha_fin, df_original3)
        df_original3 = df_clean3.sort_values(by='tiempo', ascending=False).fillna('-')
        df_original3 = df_original3.rename(
            columns={'tiempo': 'Tiempo', 'frecuencia_hz': 'Frecuencia (Hz)', 'level_dbuv_m': 'Level (dBµV/m)',
                     'bandwidth_hz': 'Bandwidth (Hz)', 'city': 'Ciudad', 'estacion': 'Estación',
                     'fecha_ingreso': 'Fecha Ingreso', 'ingreso': 'Ingreso', 'oficio': 'Oficio',
                     'fecha_oficio': 'Fecha Oficio', 'plazo': 'Plazo', 'fecha_inicio': 'Inicio Autorización',
                     'fecha_fin': 'Fin Autorización', 'tipo': 'Tipo'})
        df_clean3 = df_clean3.rename(
            columns={'tiempo': 'Tiempo', 'frecuencia_hz': 'Frecuencia (Hz)', 'level_dbuv_m': 'Level (dBµV/m)',
                     'bandwidth_hz': 'Bandwidth (Hz)', 'city': 'Ciudad', 'estacion': 'Estación',
                     'fecha_ingreso': 'Fecha Ingreso', 'ingreso': 'Ingreso', 'oficio': 'Oficio',
                     'fecha_oficio': 'Fecha Oficio', 'plazo': 'Plazo', 'fecha_inicio': 'Inicio Autorización',
                     'fecha_fin': 'Fin Autorización', 'tipo': 'Tipo'})
        df_clean3 = df_clean3.drop(
            columns=['Bandwidth (Hz)', 'Fecha Ingreso', 'Ingreso', 'Oficio', 'Fecha Oficio', 'Plazo',
                     'Inicio Autorización', 'Fin Autorización', 'Tipo'])
        df_clean3 = df_clean3.groupby(['Frecuencia (Hz)', pd.Grouper(key='Tiempo', freq='D')]).agg(
            {'Level (dBµV/m)': 'max', 'Estación': 'first'}).reset_index()

    return df_original1, df_original2, df_original3, df_clean1, df_clean2, df_clean3


def clean_data(start_date: datetime.datetime, end_date: datetime.datetime, df_primary: pd.DataFrame) -> pd.DataFrame:
    start_date_day = start_date.strftime('%Y-%m-%d')
    start_date_day = pd.to_datetime(start_date_day)
    end_date_day = end_date.strftime('%Y-%m-%d')
    end_date_day = pd.to_datetime(end_date_day)

    df_primary['tiempo'] = pd.to_datetime(df_primary['tiempo'])

    all_dates = pd.date_range(start=start_date_day, end=end_date_day, freq='D')
    existing_dates = df_primary['tiempo'].dt.normalize().unique()
    missing_dates = all_dates.difference(existing_dates)

    if not missing_dates.empty:
        # Obtener todas las combinaciones únicas de frecuencia y estación
        freq_station_combinations = df_primary[['frecuencia_hz', 'estacion']].drop_duplicates()

        # Crear un DataFrame con todas las combinaciones de fechas, frecuencias y estaciones faltantes
        df_missing_dates = pd.DataFrame([
            (t, f, s) for t in missing_dates
            for f, s in freq_station_combinations.itertuples(index=False)
        ], columns=('tiempo', 'frecuencia_hz', 'estacion'))

        df_missing_dates['frecuencia_hz'] = df_missing_dates['frecuencia_hz'].astype('float64')

        df_primary = pd.concat([df_primary, df_missing_dates], ignore_index=True)

    df_complete = df_primary
    df_complete['tiempo'] = pd.to_datetime(df_complete['tiempo'], format="%d/%m/%Y %H:%M:%S.%f")

    df_complete = df_complete[(df_complete['tiempo'] >= start_date) & (df_complete['tiempo'] <= end_date)]
    df_complete['estacion'] = df_complete['estacion'].fillna('-')
    df_complete['level_dbuv_m'] = df_complete['level_dbuv_m'].fillna(0)

    return df_complete


def analyze_dataframe(df):
    """
    Analiza el DataFrame y devuelve un diccionario con información sobre sus columnas.
    """
    analysis = {}

    # Información general
    analysis['total_rows'] = len(df)
    analysis['total_columns'] = len(df.columns)
    analysis['column_names'] = list(df.columns)

    # Análisis por columna
    for column in df.columns:
        col_analysis = {}
        col_analysis['dtype'] = str(df[column].dtype)
        col_analysis['non_null_count'] = df[column].count()
        col_analysis['null_count'] = df[column].isnull().sum()
        col_analysis['unique_values'] = df[column].nunique()

        if df[column].dtype in ['int64', 'float64']:
            col_analysis['min'] = df[column].min()
            col_analysis['max'] = df[column].max()
            col_analysis['mean'] = df[column].mean()
            col_analysis['median'] = df[column].median()
        elif df[column].dtype == 'object':
            col_analysis['sample_values'] = df[column].value_counts().head().to_dict()

        analysis[column] = col_analysis

    return analysis


@cache_page(60 * 10)
def rtv_warnings_fetch_data_from_db(request):
    """
    Fetch RTV warnings data from the database based on the selected city and date range.

    Args:
        request (django.http.HttpRequest): The HTTP request object.

    Returns:
        pandas.DataFrame: A DataFrame containing the RTV warnings data.
    """
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    city = request.GET.get('city')

    # Convert start_date and end_date to datetime objects
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    # Check if the data is already in the cache
    cache_key = f"rtv_warnings_data_{city}_{start_date}_{end_date}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    try:
        conn = get_db_connection()
        query = """
                SELECT *
                FROM rtv_operation_warnings
                WHERE ciudad = %s AND tiempo >= %s AND tiempo < %s + INTERVAL '1 DAY'
            """
        params = [city, start_date, end_date]
        df = pd.read_sql(query, conn, params=params)
        conn.close()

        # Cache the data for 10 minutes
        cache.set(cache_key, df, 60 * 10)

        return df
    except Exception as e:
        print(f"Error al recuperar datos de rtv_operation_warnings: {str(e)}")
        raise Exception(f"Error fetching RTV warnings data from the database: {str(e)}")


def customize_rtv_warnings_data(request):
    """
    Customize RTV warnings data based on selected city and date range.

    Args:
        request (django.http.HttpRequest): The HTTP request object.

    Returns:
        pd.DataFrame: Processed DataFrame with RTV warnings data.
    """
    ciudad = request.GET.get('city')
    fecha_inicio = convert_start_date(request.GET.get('start_date'))
    fecha_fin = convert_end_date(request.GET.get('end_date'))

    df_warnings = rtv_warnings_fetch_data_from_db(request)

    if df_warnings is None or df_warnings.empty:
        return pd.DataFrame()

    df_warnings['tiempo'] = pd.to_datetime(df_warnings['tiempo'])
    df_warnings = df_warnings[(df_warnings['tiempo'] >= fecha_inicio) & (df_warnings['tiempo'] <= fecha_fin)]

    # Asegúrate de que los nombres de las columnas sean consistentes
    df_warnings = df_warnings.rename(columns={
        'tiempo': 'Tiempo',
        'frecuencia_hz': 'Frecuencia (Hz)',
        'ciudad': 'Ciudad',
        'estacion': 'Estación',
        'servicio': 'Servicio',
        'analogico_digital': 'Analógico/Digital',
        'level_dbuv_m': 'Level (dBµV/m)'
    })

    # Asegúrate de que todas las columnas necesarias estén presentes
    required_columns = ['Tiempo', 'Frecuencia (Hz)', 'Estación', 'Ciudad', 'Servicio', 'Level (dBµV/m)']
    for col in required_columns:
        if col not in df_warnings.columns:
            df_warnings[col] = None

    return df_warnings


def marks_rtv_warnings_data(request):
    df_warnings = customize_rtv_warnings_data(request)

    if df_warnings is None or df_warnings.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_5_days, df_9_days, df_60_days, df_91_days = process_warnings_data(df_warnings)

    # Crear df_warnings_all
    df_warnings_all = pd.merge(
        df_5_days[['Frecuencia (Hz)', 'Estación'] + [col for col in df_5_days.columns if 'Adv.' in col]],
        df_60_days[['Frecuencia (Hz)', 'Estación'] + [col for col in df_60_days.columns if 'Adv.' in col]],
        on=['Frecuencia (Hz)', 'Estación'],
        how='outer'
    )

    # Crear df_alerts_all
    df_alerts_all = pd.merge(
        df_9_days[['Frecuencia (Hz)', 'Estación'] + [col for col in df_9_days.columns if 'Alert.' in col]],
        df_91_days[['Frecuencia (Hz)', 'Estación'] + [col for col in df_91_days.columns if 'Alert.' in col]],
        on=['Frecuencia (Hz)', 'Estación'],
        how='outer'
    )

    return df_warnings_all, df_alerts_all
