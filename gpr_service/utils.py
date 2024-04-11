import pandas as pd


def update_table(dataframe, selected_indicador, table_id):
    filtered_df = dataframe.copy()

    if selected_indicador:
        filtered_df = filtered_df[filtered_df['INDICADOR_CORTO'].isin(selected_indicador)]

    return filtered_df.to_dict('records')
