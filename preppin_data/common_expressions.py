"""Common Polar Expressions
"""

import polars as pl


def parse_str_to_bool(col_name: str, true_label: str) -> pl.Expr:
    return (
        pl.when(pl.col(col_name) == true_label)
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
    )
