import warnings
from django.conf import settings
import pandas as pd
import os
import shutil
import datetime
from functools import lru_cache
import math

# Supresión de advertencias específicas de openpyxl
warnings.simplefilter(action='ignore', category=UserWarning)


def load_environment_variables():
    return {
        'SERVER_ROUTE_GPR_2023': settings.SERVER_ROUTE_GPR_2023,
        'SERVER_ROUTE_GPR': settings.SERVER_ROUTE_GPR,
        'SERVER_ROUTE_GPR_2025': settings.SERVER_ROUTE_GPR_2025,
        'FILE_INFORMES_GPR_2023': settings.FILE_INFORMES_GPR_2023,
        'FILE_INFORMES_GPR': settings.FILE_INFORMES_GPR,
        'FILE_INFORMES_GPR_2025': settings.FILE_INFORMES_GPR_2025,
        'FILE_PACT_2024': settings.FILE_PACT_2024,
        'FILE_PACT_2025': settings.FILE_PACT_2025,
        'FILE_INDICADORES_CCDE': settings.FILE_INDICADORES_CCDE,
        'FILE_INDICADORES_CCDR': settings.FILE_INDICADORES_CCDR,
        'FILE_INDICADORES_CCDS': settings.FILE_INDICADORES_CCDS,
        'COLUMNAS_INFORMES_GPR': settings.COLUMNAS_INFORMES_GPR,
        'SERVER_ROUTE_PACT_2024': settings.SERVER_ROUTE_PACT_2024,
        'SERVER_ROUTE_PACT_2025': settings.SERVER_ROUTE_PACT_2025,
        'INDICADORES_GPR_CCDE': settings.INDICADORES_GPR_CCDE,
        'INDICADORES_GPR_CCDH': settings.INDICADORES_GPR_CCDH,
        'INDICADORES_GPR_CCDS': settings.INDICADORES_GPR_CCDS,
        'INDICADORES_GPR_CCDR': settings.INDICADORES_GPR_CCDR,
        'RUTA_CCDE_01_ENE': settings.RUTA_CCDE_01_ENE,
        'RUTA_CCDE_01_FEB': settings.RUTA_CCDE_01_FEB,
        'RUTA_CCDE_01_MAR': settings.RUTA_CCDE_01_MAR,
        'RUTA_CCDE_01_ABR': settings.RUTA_CCDE_01_ABR,
        'RUTA_CCDE_01_MAY': settings.RUTA_CCDE_01_MAY,
        'RUTA_CCDE_01_JUN': settings.RUTA_CCDE_01_JUN,
        'RUTA_CCDE_01_JUL': settings.RUTA_CCDE_01_JUL,
        'RUTA_CCDE_01_AGO': settings.RUTA_CCDE_01_AGO,
        'RUTA_CCDE_01_SEP': settings.RUTA_CCDE_01_SEP,
        'RUTA_CCDE_01_OCT': settings.RUTA_CCDE_01_OCT,
        'RUTA_CCDE_01_NOV': settings.RUTA_CCDE_01_NOV,
        'RUTA_CCDE_01_DIC': settings.RUTA_CCDE_01_DIC,
        'RUTA_CCDE_02': settings.RUTA_CCDE_02,
        'RUTA_CCDE_03': settings.RUTA_CCDE_03,
        'RUTA_CCDE_04': settings.RUTA_CCDE_04,
        'RUTA_CCDE_05': settings.RUTA_CCDE_05,
        'RUTA_CCDE_06': settings.RUTA_CCDE_06,
        'RUTA_CCDE_07': settings.RUTA_CCDE_07,
        'RUTA_CCDE_09': settings.RUTA_CCDE_09,
        'RUTA_CCDE_11': settings.RUTA_CCDE_11,
        'RUTA_CCDH_01': settings.RUTA_CCDH_01,
        'RUTA_CCDR_04': settings.RUTA_CCDR_04,
        'RUTA_CCDS_03': settings.RUTA_CCDS_03,
        'RUTA_CCDS_05': settings.RUTA_CCDS_05,
        'RUTA_CCDS_08': settings.RUTA_CCDS_08,
        'RUTA_CCDS_09': settings.RUTA_CCDS_09,
        'RUTA_CCDS_10': settings.RUTA_CCDS_10,
        'RUTA_CCDS_11': settings.RUTA_CCDS_11,
        'RUTA_CCDS_12': settings.RUTA_CCDS_12,
        'RUTA_CCDS_16': settings.RUTA_CCDS_16,
        'RUTA_CCDS_17': settings.RUTA_CCDS_17,
        'RUTA_CCDS_18': settings.RUTA_CCDS_18,
        'RUTA_CCDS_23': settings.RUTA_CCDS_23,
        'RUTA_CCDS_27': settings.RUTA_CCDS_27,
        'RUTA_CCDS_30': settings.RUTA_CCDS_30,
        'RUTA_CCDS_31': settings.RUTA_CCDS_31,
        'RUTA_CCDS_32': settings.RUTA_CCDS_32,
        'RUTA_CCDE_01_ENE_2025': settings.RUTA_CCDE_01_ENE_2025,
        'RUTA_CCDE_01_FEB_2025': settings.RUTA_CCDE_01_FEB_2025,
        'RUTA_CCDE_01_MAR_2025': settings.RUTA_CCDE_01_MAR_2025,
        'RUTA_CCDE_01_ABR_2025': settings.RUTA_CCDE_01_ABR_2025,
        'RUTA_CCDE_01_MAY_2025': settings.RUTA_CCDE_01_MAY_2025,
        'RUTA_CCDE_01_JUN_2025': settings.RUTA_CCDE_01_JUN_2025,
        'RUTA_CCDE_01_JUL_2025': settings.RUTA_CCDE_01_JUL_2025,
        'RUTA_CCDE_01_AGO_2025': settings.RUTA_CCDE_01_AGO_2025,
        'RUTA_CCDE_01_SEP_2025': settings.RUTA_CCDE_01_SEP_2025,
        'RUTA_CCDE_01_OCT_2025': settings.RUTA_CCDE_01_OCT_2025,
        'RUTA_CCDE_01_NOV_2025': settings.RUTA_CCDE_01_NOV_2025,
        'RUTA_CCDE_01_DIC_2025': settings.RUTA_CCDE_01_DIC_2025,
        'RUTA_CCDE_02_2025': settings.RUTA_CCDE_02_2025,
        'RUTA_CCDE_03_2025': settings.RUTA_CCDE_03_2025,
        'RUTA_CCDE_04_2025': settings.RUTA_CCDE_04_2025,
        'RUTA_CCDE_05_2025': settings.RUTA_CCDE_05_2025,
        'RUTA_CCDE_06_2025': settings.RUTA_CCDE_06_2025,
        'RUTA_CCDE_07_2025': settings.RUTA_CCDE_07_2025,
        'RUTA_CCDE_09_2025': settings.RUTA_CCDE_09_2025,
        'RUTA_CCDE_11_2025': settings.RUTA_CCDE_11_2025,
        'RUTA_CCDH_01_2025': settings.RUTA_CCDH_01_2025,
        'RUTA_CCDR_04_2025': settings.RUTA_CCDR_04_2025,
        'RUTA_CCDS_03_2025': settings.RUTA_CCDS_03_2025,
        'RUTA_CCDS_05_2025': settings.RUTA_CCDS_05_2025,
        'RUTA_CCDS_08_2025': settings.RUTA_CCDS_08_2025,
        'RUTA_CCDS_09_2025': settings.RUTA_CCDS_09_2025,
        'RUTA_CCDS_10_2025': settings.RUTA_CCDS_10_2025,
        'RUTA_CCDS_11_2025': settings.RUTA_CCDS_11_2025,
        'RUTA_CCDS_12_2025': settings.RUTA_CCDS_12_2025,
        'RUTA_CCDS_16_2025': settings.RUTA_CCDS_16_2025,
        'RUTA_CCDS_17_2025': settings.RUTA_CCDS_17_2025,
        'RUTA_CCDS_18_2025': settings.RUTA_CCDS_18_2025,
        'RUTA_CCDS_23_2025': settings.RUTA_CCDS_23_2025,
        'RUTA_CCDS_27_2025': settings.RUTA_CCDS_27_2025,
        'RUTA_CCDS_30_2025': settings.RUTA_CCDS_30_2025,
        'RUTA_CCDS_31_2025': settings.RUTA_CCDS_31_2025,
        'RUTA_CCDS_32_2025': settings.RUTA_CCDS_32_2025,
    }


def read_excel_data2023(server_route, file_informes, columnas_informes):
    # Read DATOS sheet
    df_datos = pd.read_excel(f'{server_route}/{file_informes}', sheet_name='DATOS')
    df_datos.loc[:, ['Nro. ACTIVIDAD', 'INDICADOR AÑO']] = df_datos[['Nro. ACTIVIDAD', 'INDICADOR AÑO']].astype('int')
    df_datos.loc[:, ['ACTIVIDAD', 'RAZÓN SOCIAL', 'NOMBRE DEL SISTEMA', 'FRECUENCIA / CANAL', 'CATEGORÍA',
                     'DOCUMENTO / ANTECEDENTE', 'FECHA ANTECEDENTE', 'ASUNTO', 'INDICADOR', 'PROVINCIA', 'LUGAR']] = \
        df_datos[
            ['ACTIVIDAD', 'RAZÓN SOCIAL', 'NOMBRE DEL SISTEMA', 'FRECUENCIA / CANAL', 'CATEGORÍA',
             'DOCUMENTO / ANTECEDENTE', 'FECHA ANTECEDENTE', 'ASUNTO', 'INDICADOR', 'PROVINCIA', 'LUGAR']].astype('str')

    # Read INFORMES sheet
    df_informes = pd.read_excel(f'{server_route}/{file_informes}', sheet_name='INFORMES', skiprows=1,
                                usecols=columnas_informes)
    df_informes.loc[:, ['Nro. INFORME', 'RESPONSABLE DE ACTIVIDAD', 'REQUIRIÓ INSPECCIÓN', 'EQUIPO UTILIZADO',
                        'OBSERVACIONES']] = df_informes[
        ['Nro. INFORME', 'RESPONSABLE DE ACTIVIDAD', 'REQUIRIÓ INSPECCIÓN', 'EQUIPO UTILIZADO',
         'OBSERVACIONES']].astype('str')
    df_informes.loc[:, 'FECHA DE INFORME'] = pd.to_datetime(df_informes['FECHA DE INFORME'])

    # Merge the dataframes
    df_final = pd.merge(df_datos, df_informes, on='Nro. ACTIVIDAD', how='outer')
    df_final.loc[:, 'RUTA'] = pd.NA
    df_final.loc[:, 'INDICADOR CORTO'] = pd.NA
    df_final.loc[:, 'INFORME_FINALIZADO_SERVIDOR'] = pd.NA
    df_final.loc[:, 'INFORME_FINALIZADO_RUTA'] = pd.NA
    df_final.loc[:, 'RUTA_SERVER'] = pd.NA
    return df_final


def read_excel_data2024(server_route, file_informes, columnas_informes):
    # Read DATOS sheet
    df_datos = pd.read_excel(f'{server_route}/{file_informes}', sheet_name='DATOS')
    df_datos['RUTA'] = df_datos['RUTA'].str.replace('\\', '/')
    df_datos['Nro INSPECCIONES'] = pd.to_numeric(df_datos['Nro INSPECCIONES'], errors='coerce')
    df_datos[['Nro. ACTIVIDAD', 'INDICADOR AÑO']] = df_datos[['Nro. ACTIVIDAD', 'INDICADOR AÑO']].astype('int')
    df_datos[
        ['ACTIVIDAD', 'RAZÓN SOCIAL', 'NOMBRE DEL SISTEMA', 'FRECUENCIA / CANAL', 'CATEGORÍA',
         'DOCUMENTO / ANTECEDENTE',
         'FECHA ANTECEDENTE', 'ASUNTO', 'INDICADOR', 'PROVINCIA', 'LUGAR', 'INDICADOR CORTO', 'RUTA']] = df_datos[
        ['ACTIVIDAD', 'RAZÓN SOCIAL', 'NOMBRE DEL SISTEMA', 'FRECUENCIA / CANAL', 'CATEGORÍA',
         'DOCUMENTO / ANTECEDENTE',
         'FECHA ANTECEDENTE', 'ASUNTO', 'INDICADOR', 'PROVINCIA', 'LUGAR', 'INDICADOR CORTO', 'RUTA']].astype('str')

    # Read INFORMES sheet
    df_informes = pd.read_excel(f'{server_route}/{file_informes}', sheet_name='INFORMES', skiprows=1,
                                usecols=columnas_informes)
    df_informes['Nro. ACTIVIDAD'] = df_informes['Nro. ACTIVIDAD'].astype('int')
    df_informes[['Nro. INFORME', 'RESPONSABLE DE ACTIVIDAD', 'REQUIRIÓ INSPECCIÓN', 'EQUIPO UTILIZADO',
                 'OBSERVACIONES']] = df_informes[
        ['Nro. INFORME', 'RESPONSABLE DE ACTIVIDAD', 'REQUIRIÓ INSPECCIÓN', 'EQUIPO UTILIZADO',
         'OBSERVACIONES']].astype('str')
    df_informes['FECHA DE INFORME'] = pd.to_datetime(df_informes['FECHA DE INFORME'])
    df_informes['FECHA DE CONTROL'] = pd.to_datetime(df_informes['FECHA DE CONTROL'])

    # Merge the dataframes
    df_final = pd.merge(df_datos, df_informes, on='Nro. ACTIVIDAD', how='outer')
    return df_final


def check_pdf_exists(pdf_files, nro_informe):
    for file in pdf_files:
        if file.startswith(str(nro_informe)):
            return 'SI'
    return 'NO'


def check_pdf_exists_in_route(row):
    if row['RUTA'] != 'nan':
        ruta = row['RUTA']
        for file in os.listdir(ruta):
            if file.startswith(str(row['Nro. INFORME'])) and file.endswith('.pdf'):
                return 'SI'
    return 'NO'


def create_ruta2_informe_df(env_vars, df_datos2023, df_datos2024):
    ruta_columns = [key for key in env_vars.keys() if key.startswith('RUTA_')]
    data = []

    for ruta_column in ruta_columns:
        ruta = env_vars[ruta_column]
        indicador_corto = ruta_column[5:12].replace('_', '-')  # Extrae los 6 caracteres que siguen después de 'RUTA_'
        if os.path.exists(ruta):
            for file in os.listdir(ruta):
                if file.endswith('.pdf'):
                    nro_informe = file[:19]  # Asume que los primeros 19 caracteres son el número de informe
                    if file[13] == '3':
                        matched_row = df_datos2023[df_datos2023['Nro. INFORME'] == str(nro_informe)]
                    else:
                        matched_row = df_datos2024[df_datos2024['Nro. INFORME'] == str(nro_informe)]
                    if not matched_row.empty:
                        matched_row = matched_row.iloc[0].to_dict()
                        matched_row['RUTA_SERVER'] = ruta
                        matched_row['INDICADOR_CORTO'] = indicador_corto
                        matched_row['INFORME_FINALIZADO_RUTA'] = 'SI'
                        data.append(matched_row)

    return pd.DataFrame(data)


def sincronizar_informes(row, server_route, pdf_files):
    if row['RUTA'] != 'nan':
        if row['INFORME_FINALIZADO_SERVIDOR'] == 'SI' and row['INFORME_FINALIZADO_RUTA'] == 'NO':
            archivo_servidor = [f for f in pdf_files if f.startswith(str(row['Nro. INFORME']))][0]
            shutil.copyfile(f'{server_route}/{archivo_servidor}', f'{row['RUTA']}/{archivo_servidor}')
        elif row['INFORME_FINALIZADO_SERVIDOR'] == 'NO' and row['INFORME_FINALIZADO_RUTA'] == 'SI':
            ruta_archivos = os.listdir(row['RUTA'])
            archivo_ruta = [f for f in ruta_archivos if f.startswith(str(row['Nro. INFORME'])) and f.endswith('.pdf')][
                0]
            shutil.copyfile(f'{row['RUTA']}/{archivo_ruta}', f'{server_route}/{archivo_ruta}')


def process_data(fecha_str='2025-03-31'):
    env_vars = load_environment_variables()

    fecha_limite = pd.to_datetime(fecha_str) + pd.Timedelta(days=11)

    df_original2023 = read_excel_data2023(env_vars['SERVER_ROUTE_GPR_2023'], env_vars['FILE_INFORMES_GPR_2023'],
                                          env_vars['COLUMNAS_INFORMES_GPR'])
    df_original2023.rename(columns={'INDICADOR AÑO': 'INDICADOR_AÑO', 'INDICADOR CORTO': 'INDICADOR_CORTO'},
                           inplace=True)

    df_original2024 = read_excel_data2024(env_vars['SERVER_ROUTE_GPR'], env_vars['FILE_INFORMES_GPR'],
                                          env_vars['COLUMNAS_INFORMES_GPR'])

    # Get the list of all .pdf files in the SERVER_ROUTE directory
    pdf_files = [f for f in os.listdir(env_vars['SERVER_ROUTE_GPR']) if f.endswith('.pdf')]

    # Apply functions to the dataframe
    df_original2024['INFORME_FINALIZADO_SERVIDOR'] = df_original2024['Nro. INFORME'].apply(
        lambda x: check_pdf_exists(pdf_files, x))
    df_original2024['INFORME_FINALIZADO_RUTA'] = df_original2024.apply(check_pdf_exists_in_route, axis=1)
    df_original2024.rename(columns={'INDICADOR AÑO': 'INDICADOR_AÑO', 'INDICADOR CORTO': 'INDICADOR_CORTO'},
                           inplace=True)
    df_original2024['RUTA_SERVER'] = pd.NA
    df_original_rutas = create_ruta2_informe_df(env_vars, df_original2023, df_original2024)
    df_final = df_original_rutas.sort_values(by='INDICADOR_CORTO')
    df_final = df_final[
        ['INDICADOR_CORTO', 'Nro. ACTIVIDAD', 'NOMBRE DEL SISTEMA', 'CATEGORÍA', 'DOCUMENTO / ANTECEDENTE',
         'RUTA_SERVER', 'Nro INSPECCIONES', 'Nro. INFORME', 'FECHA DE INFORME', 'RESPONSABLE DE ACTIVIDAD']]
    df_final = df_final[df_final['FECHA DE INFORME'] <= fecha_limite]

    return df_final


@lru_cache(maxsize=32)
def process_data_cached(fecha_str='2025-03-31'):
    return process_data(fecha_str)


"""CARGA DE DATOS PACT 2024"""


def cargar_datos1(ruta_archivo):
    return pd.ExcelFile(ruta_archivo)


def cargar_datos2(ruta_archivo):
    data = pd.read_excel(ruta_archivo)
    return data


@lru_cache(maxsize=10)
def leer_hoja_cached(xl, nombre_hoja, corto_len=7):
    df = xl.parse(nombre_hoja)
    df['INDICADOR_CORTO'] = df['INDICADOR'].str[:corto_len]
    return df


def seleccionar_columnas(df, config):
    df = df.loc[:, config['columns']]
    df.rename(columns=config['rename_columns'], inplace=True)
    df.loc[:, 'TIPO'] = df['TIPO'].str.upper()
    df_filtered = df[df['UNIDAD_ADMINISTRATIVA'] == config['UNIDAD_ADMINISTRATIVA']].iloc[:-1]
    return df_filtered


def concatenar_dataframes(dataframes):
    resultado = pd.concat(dataframes, axis=0)
    return resultado.reset_index(drop=True)


def denominadores_indicadores_dinamicos(ruta1, ruta2, ruta3):
    denominadores_CCDE = cargar_datos2(ruta1)
    denominadores_CCDR = cargar_datos2(ruta2)
    denominadores_CCDS = cargar_datos2(ruta3)
    # Set the 'INDICADOR_CORTO' column as the index for each DataFrame
    denominadores_CCDE.set_index('INDICADOR_CORTO', inplace=True)
    denominadores_CCDR.set_index('INDICADOR_CORTO', inplace=True)
    denominadores_CCDS.set_index('INDICADOR_CORTO', inplace=True)

    resultado = pd.concat([denominadores_CCDE, denominadores_CCDR, denominadores_CCDS], axis=0)
    return resultado.reset_index()


def pact_2024(fecha_str='2024-01-31'):
    env_vars = load_environment_variables()
    ruta_archivo = f'{env_vars['SERVER_ROUTE_PACT_2024']}/{env_vars['FILE_PACT_2024']}'
    ruta_discretos_CCDE = f'{env_vars['SERVER_ROUTE_PACT_2024']}/{env_vars['FILE_INDICADORES_CCDE']}'
    ruta_discretos_CCDR = f'{env_vars['SERVER_ROUTE_PACT_2024']}/{env_vars['FILE_INDICADORES_CCDR']}'
    ruta_discretos_CCDS = f'{env_vars['SERVER_ROUTE_PACT_2024']}/{env_vars['FILE_INDICADORES_CCDS']}'
    xl = cargar_datos1(ruta_archivo)
    configs = {
        'CCDE': {
            'columns': [
                "Unidad Administrativa", "INDICADOR_CORTO", "INDICADOR", "FORMULA", "Discreto / Continuo",
                "PLANIFICADA",
                "META ANUAL", "Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
                "OBSERVACIONES"
            ],
            'rename_columns': {
                "Unidad Administrativa": "UNIDAD_ADMINISTRATIVA",
                "Discreto / Continuo": "TIPO",
                "META ANUAL": "META_ANUAL"
            },
            'UNIDAD_ADMINISTRATIVA': 'CZO2'
        },
        'CCDH': {
            'columns': [
                "Unidad Administrativa", "INDICADOR_CORTO", "INDICADOR", "FORMULA", "Discreto / Continuo",
                "CANTIDAD Planificada",
                "META ANUAL", "Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
                "OBSERVACIONES"
            ],
            'rename_columns': {
                "Unidad Administrativa": "UNIDAD_ADMINISTRATIVA",
                "Discreto / Continuo": "TIPO",
                "CANTIDAD Planificada": "PLANIFICADA",
                "META ANUAL": "META_ANUAL"
            },
            'UNIDAD_ADMINISTRATIVA': 'CZO2'
        },
        'CCDR': {
            'columns': [
                "Unidad Administrativa", "INDICADOR_CORTO", "INDICADOR", "FORMULA", "DISCRETO / CONTINUO",
                "PLANIFICADA", "META ANUAL - Número de Actividades de Control (Programadas)", "Ene", "Feb", "Mar",
                "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic", "OBSERVACIONES"
            ],
            'rename_columns': {
                "Unidad Administrativa": "UNIDAD_ADMINISTRATIVA",
                "DISCRETO / CONTINUO": "TIPO",
                "META ANUAL - Número de Actividades de Control (Programadas)": "META_ANUAL"
            },
            'UNIDAD_ADMINISTRATIVA': 'CZO2'
        },
        'CCDS': {
            'columns': [
                "Unidad Administrativa", "INDICADOR_CORTO", "INDICADOR", "FORMULA", "Discreto / Continuo",
                "CANTIDAD Planificada / Estimada", "META ANUAL - Número de Actividades de Control (Programadas)", "Ene",
                "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic", "OBSERVACIONES"
            ],
            'rename_columns': {
                "Unidad Administrativa": "UNIDAD_ADMINISTRATIVA",
                "Discreto / Continuo": "TIPO",
                "CANTIDAD Planificada / Estimada": "PLANIFICADA",
                "META ANUAL - Número de Actividades de Control (Programadas)": "META_ANUAL"
            },
            'UNIDAD_ADMINISTRATIVA': 'CZO2'
        }
        # Agrega aquí configuraciones adicionales para otras hojas
    }
    dataframes = []
    for hoja, config in configs.items():
        df = leer_hoja_cached(xl, hoja)
        df = seleccionar_columnas(df, config)
        dataframes.append(df)
    resultado = concatenar_dataframes(dataframes)
    denom_ind_din = denominadores_indicadores_dinamicos(ruta_discretos_CCDE, ruta_discretos_CCDR, ruta_discretos_CCDS)
    resultado = pd.merge(resultado, denom_ind_din, how='outer', on=['UNIDAD_ADMINISTRATIVA', 'INDICADOR_CORTO'])
    return resultado


def procesar_mes_con_fecha(dataframe, fecha_str):
    fecha = datetime.datetime.strptime(fecha_str, '%Y-%m-%d')
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    indice_mes = fecha.month - 1
    meses_evaluar = meses[:indice_mes + 1]

    def calcular_cumplir(row):
        if row['TIPO'] == 'CONTINUO':
            valores_no_nulos = row[meses_evaluar].dropna()
            return valores_no_nulos.sum() if not valores_no_nulos.empty else 0.0
        else:  # Para tipo DISCRETO
            return sum(
                row[f"{mes}_den"] for mes in meses_evaluar if pd.notnull(row[mes]) and pd.notnull(row[f"{mes}_den"]))

    dataframe['CUMPLIR'] = dataframe.apply(calcular_cumplir, axis=1)

    def calcular_cumplir_meta(row):
        if row['TIPO'] == 'CONTINUO':
            valores_no_nulos = row[meses_evaluar].dropna()
            return valores_no_nulos.sum() * row['META_ANUAL'] if not valores_no_nulos.empty else 0.0
        else:  # Para tipo DISCRETO
            if row['INDICADOR_CORTO'] in ['CCDR-04']:
                return sum(row[f"{mes}_den"] * row['META_ANUAL'] for mes in meses_evaluar if
                           pd.notnull(row[mes]) and pd.notnull(row[f"{mes}_den"]))
            else:
                return sum(row[mes] * row[f"{mes}_den"] for mes in meses_evaluar if
                           pd.notnull(row[mes]) and pd.notnull(row[f"{mes}_den"]))

    dataframe['CUMPLIR_META'] = dataframe.apply(calcular_cumplir_meta, axis=1).apply(math.ceil)

    return dataframe


def actualizar_planificada(df):
    # # Lista de INDICADOR_CORTO que requieren actualización siempre
    # indicadores_actualizar = ['CCDE-05', 'CCDE-06', 'CCDE-07', 'CCDE-09']

    # Verificar si 'PLANIFICADA' existe en el DataFrame
    if 'PLANIFICADA' not in df.columns:
        raise ValueError("La columna 'PLANIFICADA' no está presente en el DataFrame.")

    # Iterar sobre las filas del DataFrame
    for index, row in df.iterrows():
        # Verificar si el INDICADOR_CORTO de la fila es uno de los que siempre se deben actualizar
        if pd.isnull(row['PLANIFICADA']):
            if pd.notnull(row['CUMPLIR']) and pd.notnull(row['META_ANUAL']) and row['META_ANUAL'] != 0:
                # Calcular el nuevo valor de PLANIFICADA
                nuevo_valor = row['CUMPLIR']
                # Actualizar el valor en el DataFrame
                df.at[index, 'PLANIFICADA'] = nuevo_valor
    # Crear la columna PLANIFICADA_META como multiplicación de PLANIFICADA por META_ANUAL
    df['PLANIFICADA_META'] = (df['PLANIFICADA'] * df['META_ANUAL']).apply(math.ceil)

    return df


"""REVISIÓN VERIFICABLES A LA FECHA"""


def preparar_datos(df):
    # Convertir la columna 'FECHA DE INFORME' a tipo datetime
    df['FECHA DE INFORME'] = pd.to_datetime(df['FECHA DE INFORME'])

    # Crear la columna 'MES' extrayendo el nombre del mes en formato abreviado español
    meses_abreviados = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
    }
    df['MES'] = df['FECHA DE INFORME'].dt.month.map(meses_abreviados)
    return df


def informes_acumulados_por_fecha(df, fecha_final):
    # Convertir la cadena de fecha de entrada a tipo datetime si es necesario
    if isinstance(fecha_final, str):
        fecha_final = pd.to_datetime(fecha_final)

    # Filtrar los datos para incluir solo hasta la fecha especificada
    df_filtrado = df[df['FECHA DE INFORME'] <= fecha_final]

    # Agrupar por 'INDICADOR_CORTO' y contar las filas por grupo y sumar 'Nro. INSPECCIONES'
    df_resultado = df_filtrado.groupby('INDICADOR_CORTO').agg(CANTIDAD_VERIFICABLES=('INDICADOR_CORTO', 'size'),
                                                              Nro_INSPECCIONES=(
                                                                  'Nro INSPECCIONES', 'sum')).reset_index()
    return df_resultado


def verificables(data, fecha_str='2024-01-31'):
    data_preparada = preparar_datos(data)
    fecha_con_dias_adicionales = pd.to_datetime(fecha_str) + pd.Timedelta(days=11)
    resultado = informes_acumulados_por_fecha(data_preparada, fecha_con_dias_adicionales)
    return resultado


def pac_verificables(df1, df2):
    resultado = pd.merge(df1, df2, how='outer', on='INDICADOR_CORTO')
    return resultado.round(2)
