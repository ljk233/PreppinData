"""2023: Week 5 - DSB Ranking

Notes
-----
The recommended solution contains an error due to grouping the data by month
name. However, a single transaction in January 2024 causes the January output
to include transactions from both January 2023 and January 2024. Consequently,
the correct solution should produce 37 rows instead of 36 rows to accurately
reflect the data.


Inputs
------
- __input/2023/PD 2023 Wk 1 Input.csv

Outputs
-------
- output/2023/wk05_new_customer.ndjson
"""

import polars as pl

from .challenge01 import load_transactions_data, preprocess_transactions_data


def solve(pd_input_wk1_fsrc: str) -> pl.DataFrame:
    """Solve challenge 5 of Preppin' Data 2023.

    Parameters
    ----------
    pd_input_wk1_fsrc : str
        Filepath of the input CSV file for Week 1.

    Returns
    -------
    pl.DataFrame
        DataFrame containing summary data on the transaction values and
        bank's rank compared to other similar banks.


    Notes
    -----
    This function loads and preprocesses transaction data for Week 1 of
    Preppin' Data 2023. It then generates a monthly summary of total transaction
    values and ranks them. The output DataFrame includes the mean transaction
    value per rank and the mean rank per bank.
    """

    # Load the data
    transaction = load_transactions_data(pd_input_wk1_fsrc)

    # Preprocess the data
    pre_transaction = transaction.pipe(preprocess_transactions_data)

    return pre_transaction.pipe(view_ranked_monthly_summary).collect()


def view_ranked_monthly_summary(pre_transaction: pl.LazyFrame) -> pl.LazyFrame:
    """Generate a monthly summary of the ranked transaction values.

    Parameters
    ----------
    pre_transaction : pl.LazyFrame
        LazyFrame representing the preprocessed transaction data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the monthly summary of transaction values,
        including ranks, mean transaction value per rank, and mean rank
        per bank.
    """

    mean_value_per_rank_expr = pl.mean("total_transaction_value").over("rank")

    mean_rank_per_bank_expr = pl.mean("rank").over("bank")

    return (
        pre_transaction.pipe(aggregate_data)
        .pipe(rank_aggregate_data)
        .with_columns(
            mean_transaction_value_per_rank=mean_value_per_rank_expr,
            mean_rank_per_bank=mean_rank_per_bank_expr,
        )
    )


def aggregate_data(pre_transaction: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate the total transactions over bank and month.

    Parameters
    ----------
    pre_transaction : pl.LazyFrame
        LazyFrame representing the preprocessed transaction data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the aggregated total transaction values over
        bank and month.
    """

    transactions_in_expr = pl.col("created_on").dt.strftime("%B-%Y")

    return pre_transaction.group_by("bank", transactions_in=transactions_in_expr).agg(
        total_transaction_value=pl.sum("value")
    )


def rank_aggregate_data(agg_data: pl.DataFrame) -> pl.DataFrame:
    """Rank the total transaction value for each bank by month.

    Parameters
    ----------
    agg_data : pl.LazyFrame
        LazyFrame representing the aggregated total transaction values over
        bank and month.

    Returns
    -------
    pl.DataFrame
        LazyFrame representing the rank of total transaction values for
        each bank per month.
    """

    rank_total_transaction_val_expr = (
        pl.col("total_transaction_value")
        .rank("min", descending=True)
        .over("transactions_in")
    )

    return agg_data.with_columns(rank=rank_total_transaction_val_expr)
