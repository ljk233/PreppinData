"""Common Polar Expressions
"""

from typing import Any

import polars as pl


def parse_date(col_name, format: str, **kwargs) -> pl.Expr:
    """"""
    return pl.col(col_name).str.to_date(format, **kwargs)


def if_else(
    col_name: str,
    true_val: Any,
    true_label: Any,
    false_label: Any,
    operator: str = "==",
) -> pl.Expr:
    """
    Return an expression representing an if-else conditional logic for a
    column in Polars DataFrame.

    Parameters
    ----------
    col_name : str
        The name of the column to apply the conditional logic to.
    true_val : Any
        The value to compare against in the conditional logic.
    true_label : Any
        The label to assign to rows where the condition is true.
    false_label : Any
        The label to assign to rows where the condition is false.

    Returns
    -------
    pl.Expr
        An expression representing the if-else conditional logic.

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
    pred_expr = get_predicate_expr(col_name, operator, true_val)

    return pl.when(pred_expr).then(pl.lit(true_label)).otherwise(pl.lit(false_label))


def get_predicate_expr(col_name: str, operator: str, true_val: Any) -> pl.Expr:
    """"""
    col_expr = pl.col(col_name)
    match operator:
        case "==":
            return col_expr == true_val
        case "<":
            return col_expr < true_val
        case "<=":
            return col_expr <= true_val
        case ">=":
            return col_expr >= true_val
        case ">":
            return col_expr > true_val
        case "in":
            return col_expr.is_in(set(true_val))
        case _:
            raise NotImplementedError(f"Operator {operator} is not yet implemented.")


def approx_years_between(start_date_col: str, end_date_col: str) -> pl.Expr:
    """Return the approximate years between the start and end date.

    Parameters
    ----------
    start_date_col : str
        The name of the column containing the start dates.
    end_date_col : str
        The name of the column containing the end dates.

    Returns
    -------
    pl.Expr
        A Polars expression representing the approximate difference in years.

    Examples
    --------
    >>> import polars as pl
    >>>
    >>> # Example usage
    >>> start_date = ["2020-01-01", "2023-07-01"]
    >>> end_date = ["2021-01-01", "2025-06-30"]
    >>>
    >>> df = pl.DataFrame({
    ...     "start_date": start_date,
    ...     "end_date": end_date
    ... })
    >>>
    >>> approx_years_expr = approx_years_between("start_date", "end_date")
    >>> result = df.select([
    ...     approx_years_expr.alias("approx_years_between")
    ... ])
    >>>
    >>> print(result)
    shape: (2, 1)
    ┌─────────────────────┐
    │ approx_years_between│
    │ int                 │
    │ ---                 │
    │ 1.0                 │
    │ 1.4997241665518486  │
    └─────────────────────┘
    """
    duration = pl.col(end_date_col) - pl.col(start_date_col)

    return duration.dt.total_days() / 365.25


def approx_months_between(start_date_col: str, end_date_col: str) -> pl.Expr:
    """Return the approximate months between the start and end date.

    Parameters
    ----------
    start_date_col : str
        The name of the column containing the start dates.
    end_date_col : str
        The name of the column containing the end dates.

    Returns
    -------
    pl.Expr
        A Polars expression representing the approximate difference in months.

    Examples
    --------
    >>> import polars as pl
    >>>
    >>> # Example usage
    >>> start_date = ["2020-01-01", "2023-07-01"]
    >>> end_date = ["2021-01-01", "2025-06-30"]
    >>>
    >>> df = pl.DataFrame({
    ...     "start_date": start_date,
    ...     "end_date": end_date
    ... })
    >>>
    >>> approx_month_expr = approx_month_between("start_date", "end_date")
    >>> result = df.select([
    ...     approx_month_expr.alias("approx_months_between")
    ... ])
    >>>
    >>> print(result)
    shape: (2, 1)
    ┌──────────────────────┐
    │ approx_months_between│
    │ float64              │
    │ ---                  │
    │ 12.0                 │
    │ 18.329394616840275   │
    └──────────────────────┘
    """
    duration = pl.col(end_date_col) - pl.col(start_date_col)

    return 12 * duration.dt.total_days() / 365.25
