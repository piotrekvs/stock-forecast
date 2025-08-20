import pandas as pd

def localize_dataframe_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.tz_localize(None)

def get_df_with_columns(df: pd.DataFrame, column_names: list) -> pd.DataFrame:
    missing = [col for col in column_names if col not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} do not exist in the DataFrame.")
    return df[column_names].copy()
