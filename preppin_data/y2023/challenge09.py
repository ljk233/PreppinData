"""2023: Week 9 - Customer Bank Statements

See solution output at "output/2023/wk09_running_balance_summary.ndjson".
"""

import polars as pl

from .challenge07 import (
    preprocess_account_info,
    preprocess_transaction_detail,
    preprocess_transaction_path,
)


def solve(
    account_info_fsrc: str,
    transaction_detail_fsrc: str,
    transaction_path_fsrc: str,
) -> pl.LazyFrame:
    """Solve challenge 9 of Preppin' Data 2023."""
    # Preprocess the data
    pre_account_info = preprocess_account_info(account_info_fsrc)
    pre_transaction_detail = preprocess_transaction_detail(transaction_detail_fsrc)
    pre_transaction_path = preprocess_transaction_path(transaction_path_fsrc)

    # Further processing and harmonization
    account_info = pre_account_info.unique(["account_number"]).select(
        "account_number",
        "balance_taken_on",
        pl.col("balance").alias("transaction_value"),
    )

    daily_transaction_flow = (
        pre_transaction_path.melt(
            id_vars="transaction_id", variable_name="edge", value_name="account_number"
        )
        .join(pre_transaction_detail, on="transaction_id")
        .filter(pl.col("is_cancelled").not_())
        .with_columns(sign_edge("edge").alias("multiplier"))
        .select(
            "account_number",
            pl.col("created_on").alias("balance_taken_on"),
            pl.col("transaction_value") * pl.col("multiplier"),
        )
    )

    return (
        pl.concat([account_info, daily_transaction_flow])
        .sort("account_number", "balance_taken_on", "transaction_value")
        .with_columns(
            pl.cum_sum("transaction_value").over("account_number").alias("balance")
        )
        .with_columns(
            mask_transaction_value("transaction_value", "balance_taken_on").alias(
                "transaction_value"
            )
        )
        .select("account_number", "balance_taken_on", "transaction_value", "balance")
    )


def sign_edge(edge_col: str) -> pl.Expr:
    """Return a signed unit integer representation of the edge.

    If the transaction represents the inbound edge, then 1 is returned.
    Otherwise if the transaction represents the outbound edge, then 1 is returned.
    Otherwise, return 0.
    """
    return (
        pl.when(pl.col(edge_col).str.starts_with("inbound"))
        .then(pl.lit(1))
        .when(pl.col(edge_col).str.starts_with("outbound"))
        .then(pl.lit(-1))
        .otherwise(pl.lit(None))
    )


def mask_transaction_value(transaction_value: str, balance_taken_on: str):
    """Mask the transaction value if the balance was taken in January."""
    return (
        pl.when(pl.col(balance_taken_on).dt.month() == 1)
        .then(pl.lit(None))
        .otherwise(transaction_value)
    )
