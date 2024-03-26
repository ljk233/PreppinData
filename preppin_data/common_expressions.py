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
