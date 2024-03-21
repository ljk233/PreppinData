"""
"""

import re
from datetime import date

import polars as pl

from .challenge09 import solve as solve_challenge09


def solve(
    account_info_fsrc: str,
    transaction_detail_fsrc: str,
    transaction_path_fsrc: str,
    balance_on: str = "2023-02-01",
) -> pl.LazyFrame:
    """"""
    # Parse the date string
    balance_on_date = parse_date_str(balance_on)

    # Return the solution to challenge 9
    running_balance_summary = solve_challenge09(
        account_info_fsrc, transaction_detail_fsrc, transaction_path_fsrc
    )

    # Aggregate the data so we have one row per account
    daily_balance_summary = running_balance_summary.group_by(
        "account_number", "balance_taken_on"
    ).agg(pl.sum("transaction_value"), pl.last("balance"))

    # Scaffold the data
    unique_accounts = daily_balance_summary.select("account_number").unique()

    st_day = parse_date_str("2023-01-31")
    end_day = parse_date_str("2023-02-14")
    balance_taken_on = pl.LazyFrame().select(
        pl.date_range(st_day, end_day).alias("balance_taken_on")
    )

    return (
        unique_accounts.join(balance_taken_on, how="cross")
        .join(
            daily_balance_summary, on=["account_number", "balance_taken_on"], how="left"
        )
        .with_columns(pl.col("balance").fill_null(strategy="forward"))
        .filter(pl.col("balance_taken_on") == balance_on_date)
    )


def parse_date_str(s: str) -> pl.Expr:
    return date(*[int(d) for d in re.findall(r"\d+", s)])
