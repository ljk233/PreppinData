"""Common Polar Expressions
"""

from polars import col, Expr


def alias(from_col_name: str, to_col_name: str) -> Expr:
    """Wrapper for pl.col(from_col_name).alias(to_col_name)."""
    return col(from_col_name).alias(to_col_name)


def parse_date_str(col_name: str, format: str) -> Expr:
    """Wrapper for pl.col(from_col_name).str.to_date(format)."""
    return col(col_name).str.to_date(format)
