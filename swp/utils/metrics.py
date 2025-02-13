import pandas as pd


def calc_accuracy(df: pd.DataFrame, error_condition, total_condition) -> float:
    """
    Compute accuracy as 1 - (number of errors / total items) for a given condition.
    """
    total = df.loc[total_condition].shape[0]
    errors = df.loc[error_condition].shape[0]
    return 1 - errors / total
