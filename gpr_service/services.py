from django.conf import settings
import pandas as pd
import os
import shutil


def load_environment_variables():
    return {
        'SERVER_ROUTE_GPR_2023': settings.SERVER_ROUTE_GPR_2023,
        'SERVER_ROUTE_GPR': settings.SERVER_ROUTE_GPR,
        'FILE_INFORMES_GPR_2023': settings.FILE_INFORMES_GPR_2023,
        'FILE_INFORMES_GPR': settings.FILE_INFORMES_GPR,
        'COLUMNAS_INFORMES_GPR': settings.COLUMNAS_INFORMES_GPR,
        'INDICADORES_GPR_CCDE': settings.INDICADORES_GPR_CCDE,
        'INDICADORES_GPR_CCDH': settings.INDICADORES_GPR_CCDH,
        'INDICADORES_GPR_CCDS': settings.INDICADORES_GPR_CCDS,
        'INDICADORES_GPR_CCDR': settings.INDICADORES_GPR_CCDR,
        'RUTA_CCDE_01_ENE': settings.RUTA_CCDE_01_ENE,
        'RUTA_CCDE_01_FEB': settings.RUTA_CCDE_01_FEB,
        'RUTA_CCDE_01_MAR': settings.RUTA_CCDE_01_MAR,
        'RUTA_CCDE_02': settings.RUTA_CCDE_02,
        'RUTA_CCDE_03': settings.RUTA_CCDE_03,
        'RUTA_CCDE_04': settings.RUTA_CCDE_04,
        'RUTA_CCDE_05': settings.RUTA_CCDE_05,
        'RUTA_CCDE_06': settings.RUTA_CCDE_06,
        'RUTA_CCDE_07': settings.RUTA_CCDE_07,
        'RUTA_CCDE_08': settings.RUTA_CCDE_08,
        'RUTA_CCDE_09': settings.RUTA_CCDE_09,
        'RUTA_CCDE_10': settings.RUTA_CCDE_10,
        'RUTA_CCDE_11': settings.RUTA_CCDE_11,
        'RUTA_CCDH_01': settings.RUTA_CCDH_01,
        'RUTA_CCDS_01': settings.RUTA_CCDS_01,
        'RUTA_CCDS_03': settings.RUTA_CCDS_03,
        'RUTA_CCDS_05': settings.RUTA_CCDS_05,
        'RUTA_CCDS_08': settings.RUTA_CCDS_08,
        'RUTA_CCDS_09': settings.RUTA_CCDS_09,
        'RUTA_CCDS_10': settings.RUTA_CCDS_10,
        'RUTA_CCDS_11': settings.RUTA_CCDS_11,
        'RUTA_CCDS_12': settings.RUTA_CCDS_12,
        'RUTA_CCDS_13': settings.RUTA_CCDS_13,
        'RUTA_CCDS_16': settings.RUTA_CCDS_16,
        'RUTA_CCDS_17': settings.RUTA_CCDS_17,
        'RUTA_CCDS_18': settings.RUTA_CCDS_18,
        'RUTA_CCDS_23': settings.RUTA_CCDS_23,
        'RUTA_CCDS_27': settings.RUTA_CCDS_27,
        'RUTA_CCDS_30': settings.RUTA_CCDS_30,
        'RUTA_CCDS_31': settings.RUTA_CCDS_31,
        'RUTA_CCDS_32': settings.RUTA_CCDS_32,
        'RUTA_CCDR_01': settings.RUTA_CCDR_01,
        'RUTA_CCDR_04': settings.RUTA_CCDR_04,
        'RUTA_CCDR_06': settings.RUTA_CCDR_06,
    }


def read_excel_data2023(server_route, file_informes, columnas_informes):
    # Read DATOS sheet
    df_datos = pd.read_excel(f'{server_route}/{file_informes}', sheet_name='DATOS')
    df_datos[['Nro. ACTIVIDAD', 'INDICADOR AÑO']] = df_datos[['Nro. ACTIVIDAD', 'INDICADOR AÑO']].astype('int')
    df_datos[
        ['ACTIVIDAD', 'RAZÓN SOCIAL', 'NOMBRE DEL SISTEMA', 'FRECUENCIA / CANAL', 'CATEGORÍA',
         'DOCUMENTO / ANTECEDENTE',
         'FECHA ANTECEDENTE', 'ASUNTO', 'INDICADOR', 'PROVINCIA', 'LUGAR']] = df_datos[
        ['ACTIVIDAD', 'RAZÓN SOCIAL', 'NOMBRE DEL SISTEMA', 'FRECUENCIA / CANAL', 'CATEGORÍA',
         'DOCUMENTO / ANTECEDENTE',
         'FECHA ANTECEDENTE', 'ASUNTO', 'INDICADOR', 'PROVINCIA', 'LUGAR']].astype('str')

    # Read INFORMES sheet
    df_informes = pd.read_excel(f'{server_route}/{file_informes}', sheet_name='INFORMES', skiprows=1,
                                usecols=columnas_informes)
    df_informes[['Nro. INFORME', 'RESPONSABLE DE ACTIVIDAD', 'REQUIRIÓ INSPECCIÓN', 'EQUIPO UTILIZADO',
                 'OBSERVACIONES']] = df_informes[
        ['Nro. INFORME', 'RESPONSABLE DE ACTIVIDAD', 'REQUIRIÓ INSPECCIÓN', 'EQUIPO UTILIZADO',
         'OBSERVACIONES']].astype('str')
    df_informes['FECHA DE INFORME'] = pd.to_datetime(df_informes['FECHA DE INFORME'])

    # Merge the dataframes
    df_final = pd.merge(df_datos, df_informes, on='Nro. ACTIVIDAD', how='outer')
    df_final['RUTA'] = pd.NA
    df_final['INDICADOR CORTO'] = pd.NA
    df_final['INFORME_FINALIZADO_SERVIDOR'] = pd.NA
    df_final['INFORME_FINALIZADO_RUTA'] = pd.NA
    df_final['RUTA_SERVER'] = pd.NA
    return df_final


def read_excel_data2024(server_route, file_informes, columnas_informes):
    # Read DATOS sheet
    df_datos = pd.read_excel(f'{server_route}/{file_informes}', sheet_name='DATOS')
    df_datos['RUTA'] = df_datos['RUTA'].str.replace('\\', '/')
    df_datos['Nro INSPECCIONES'] = df_datos['Nro INSPECCIONES'].fillna('')
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


def process_data():
    env_vars = load_environment_variables()

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
    # df_final = df_original_rutas.query("INDICADOR_CORTO != 'Ninguno' and RUTA_SERVER =='SI'")
    df_final = df_original_rutas.sort_values(by='INDICADOR_CORTO')
    df_final = df_final[
        ['INDICADOR_CORTO', 'INDICADOR', 'Nro. ACTIVIDAD', 'NOMBRE DEL SISTEMA', 'CATEGORÍA', 'DOCUMENTO / ANTECEDENTE',
         'FECHA ANTECEDENTE', 'RUTA_SERVER', 'Nro INSPECCIONES', 'Nro. INFORME', 'FECHA DE INFORME',
         'RESPONSABLE DE ACTIVIDAD']]

    return df_final
