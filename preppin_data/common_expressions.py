"""Common Polar Expressions
"""

import polars as pl


def parse_str_to_bool(col_name: str, true_label: str, col_alias: str) -> pl.Expr:
    """Convert string values in a column to boolean values based on a specified true label.

    Parameters
    ----------
    col_name : str
        The name of the column containing string values to be converted.
    true_label : str
        The label in the column indicating true values.
    col_alias : str
        The alias to assign to the resulting boolean column.

    Returns
    -------
    pl.Expr
        A Polars expression representing the transformation.

    Notes
    -----
    This function creates a Polars expression that evaluates to True for
    rows where the value in the specified column matches the true label,
    and False otherwise. The resulting boolean olumn is given the specified alias.
    """
    return (
        pl.when(pl.col(col_name) == true_label)
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias(col_alias)
    )
