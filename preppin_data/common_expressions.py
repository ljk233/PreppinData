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


def if_else(
    col_name: str,
    true_val: str,
    true_label: str,
    false_label: str,
) -> pl.Expr:
    """
    Return an expression representing an if-else conditional logic for a
    column in Polars DataFrame.

    Parameters
    ----------
    col_name : str
        The name of the column to apply the conditional logic to.
    true_val : str
        The value to compare against in the conditional logic.
    true_label : str
        The label to assign to rows where the condition is true.
    false_label : str
        The label to assign to rows where the condition is false.

    Returns
    -------
    pl.Expr
        An expression representing the if-else conditional logic.

    Raises
    ------
    ValueError
        If an invalid operator is provided or if true_val is not a string, or if true_label or false_label are not strings.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': ['foo', 'bar', 'baz', 'foo', 'bar']
    ... })
    >>> condition_expr = if_else('A', '3', 'is_three', 'is_not_three')
    >>> result = df.select(condition_expr.alias('result'))
    >>> print(result)
    shape: (5, 1)
    ┌──────────────┐
    │ result       │
    │ ---          │
    │ str          │
    ╞══════════════╡
    │ is_not_three │
    │ is_not_three │
    │ is_three     │
    │ is_not_three │
    │ is_not_three │
    └──────────────┘
    """
    if not isinstance(true_val, str):
        raise ValueError("True value must be a string.")

    if not isinstance(true_label, str) or not isinstance(false_label, str):
        raise ValueError("Labels must be strings.")

    return (
        pl.when(pl.col(col_name) == true_val)
        .then(pl.lit(true_label))
        .otherwise(pl.lit(false_label))
    )
