import pandas as pd
from ast import literal_eval


def save_df(df, path):
    df.to_csv(path, index=False)


def get_list_type_col_names(df):
    col_names = []
    for col_name in df.columns:
        if isinstance(df[col_name][0], str) and df[col_name][0].startswith('[') and df[col_name][0].endswith(']'):
            col_names.append(col_name)
    return col_names


def safe_literal_eval(string):
    try:
        return literal_eval(string)
    except (SyntaxError, ValueError):
        pass


def get_list_converters(df_path):
    df = pd.read_csv(df_path)
    return {col_name: safe_literal_eval for col_name in get_list_type_col_names(df)}