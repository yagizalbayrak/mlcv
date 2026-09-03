import numpy as np
import pandas as pd
import typing

"""
Please add the remap dictionaries for the nametype and fall columns here. 
Now these are not complete, you need check the csv file and add more in the dictionary here.
For example, 'Vaid' should be 'Valid', 'Relic' should be 'Relict', etc.
"""
nametype_remap = {
    "Valid": "Valid",
    "Relict": "Relict",
    "Vaid": "Valid",
    "Relic": "Relict",
    "Valud": "Valid",
    "Reluct": "Relict",
    "Relit": "Relict",
    "Vald": "Valid",
    "Vlid": "Valid",
}
fall_remap = {
    "Fell": "Fell",
    "Found": "Found",
    "Fel": "Fell",
    "Feel": "Fell",
    "Foumd": "Found",
    "Feell": "Fell",
    "Fll": "Fell",
    "Fond": "Found",
    "Fund": "Found",
    "Foind": "Found",
    "Fiund": "Found",
}


def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    This function drop a given column (col). For each row:
    - Drop values from the column (col)

    Args:
        - df (pd.DataFrame): dataframe to have the columns removed
        - col (str): column to be removed

    Returns:
        - df (pd.DataFrame): dataframe without the given column
    """
    df = df.drop(columns=[col])

    return df


def remap_values(df: pd.DataFrame, col: str, remap_dict: dict) -> pd.DataFrame:
    """
    This function should remap all values from a given column (col) using a remap_dict (should be defined first).
    For each row:
    - Replace the column (col) value the mapped value on dict (rempap_dict)
    Args:
        - df (pd.DataFrame): dataframe to have the values remapped
        - col (str): column that the values will be remapped
        - remap_dict (dict): dictionary mapping the changes ('old_value': 'new_value')

    Returns:
        - df (pd.DataFrame): dataframe with all the values from column (col) remapped according to remap_dict
    """
    df[col] = df[col].map(remap_dict)

    return df


def categorical_to_num(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    This function should list all the categories within the given column (col) and
    assign a category id for each individual category, then for each row:
    - Replace the category value with the respective category id number

    You are recommended to convert the categories to numbers in order of first appearance in the column.

    Args:
        - df (pd.DataFrame): dataframe to change the categorical data to numerical
        - col (str): column to be changed

    Returns:
        - df (pd.DataFrame): dataframe with the categorical class replaced with categories ids numbers
    """
    categories = df[col].unique()
    cat_to_num = {cat: idx for idx, cat in enumerate(categories)}
    df[col] = df[col].map(cat_to_num)

    return df


def remove_row_outside_range(df: pd.DataFrame, col: str, min_val: float, max_val: float) -> pd.DataFrame:
    """
    This function should remove rows where the column (col) value is not inside a given inclusive range (min_val, max_val).
    For each row:
    - Check if the column (col) attribute is inside the range:
        - if is not in range: drop the row
        - if is in range: keep the row

    Args:
        - df (pd.DataFrame): dataframe that will have the rows removed
        - col (str): column to check the value (and decide if the row will be removed)
        - min_val (float): min allowed value for the column (values below this will be removed)
        - max_val (float): max allowed value for the column (values above this will be removed)

    Returns:
        - df (pd.DataFrame): dataframe without the rows where column (col) is not within range (min_val, max_val)
    """
    df = df[(df[col] >= min_val) & (df[col] <= max_val)]

    return df


def replace_nan_with_mean_class(df: pd.DataFrame, col: str, refcol: str) -> pd.DataFrame:
    """
    This function should go over the values in column (col) where the value is NaN, then for each row:
    - Get its refcol value (class)
    - Calculate the mean value of col for the corresponding refcol class
    - Replace the row nan value with this mean value

    Args:
        - df (pd.DataFrame): dataframe to replace nans with mean class
        - col (str): column to replace the NaNs
        - refcol (str): column to be used as reference to calculate the mean value of the col value

    Returns:
        - df (pd.DataFrame): dataframe with the mass NaNs replaced with mean mass of the recclass
    """
    class_means = df.groupby(refcol)[col].transform('mean')
    df[col] = df[col].fillna(class_means)

    return df


def remove_nan_rows(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    This function should list all the nan values within the given column then for each row:
    - Remove the current row if the column (col) is NaN

    Args:
        - df (pd.DataFrame): dataframe to have the NaNs drop
        - col (str): column in which the NaNs will be removed

    Returns:
        - df (pd.DataFrame): dataframe without the rows where the column (col) contains NaN
    """
    df = df.dropna(subset=[col])

    return df
