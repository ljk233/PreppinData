"""2023: Week 3 - Targets for DSB

Inputs
------
- __input/2023/PD 2023 Wk 1 Input.csv
- __input/2023/Targets.csv

Outputs
-------
- output/2023/wk03_dsb_variance_to_target.ndjson
"""

import polars as pl

from .challenge01 import (
    load_transactions_data,
    preprocess_transactions_data,
    aggregate_total_value,
)


def solve(pd_input_wk1_fsrc: str, targets_fsrc: str) -> pl.DataFrame:
    """Solve challenge 3 of Preppin' Data 2023.

    Parameters
    ----------
    pd_input_wk1_fsrc : str
        Filepath of the input CSV file for Week 1.
    targets_fsrc : str
        Filepath of the input CSV file containing targets data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing variance to target for DSB transactions.

    Notes
    -----
    This function reads transaction data and targets data from CSV files,
    preprocesses them, calculates the variance to target for DSB transactions,
    and returns the result.
    """

    # Load the data
    transaction = load_transactions_data(pd_input_wk1_fsrc)

    target = load_target_data(targets_fsrc)

    # Preprocess the data
    pre_transaction = transaction.pipe(preprocess_transactions_data)

    pre_target = target.pipe(preprocess_target_data)

    return view_dsb_variance_to_target(pre_transaction, pre_target).collect()


def load_target_data(fsrc: str) -> pl.LazyFrame:
    """Load data from a CSV file.

    Parameters
    ----------
    fsrc : str
        Filepath of the input CSV file to load.

    Returns
    -------
    pl.DataFrame
        LazyFrame representing the loaded data.
    """
    data = pl.scan_csv(fsrc)

    return data


def preprocess_target_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocess the target data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the target data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed target data.

    Notes
    -----
    Primary key is {transaction_method, quarter}
    """
    return data.pipe(reshape_target_data).pipe(clean_target_data)


def reshape_target_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Reshape the target data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the target data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the reshaped target data.
    """
    return data.melt(
        id_vars="Online or In-Person",
        variable_name="quarter",
        value_name="target",
    )


def clean_target_data(reshaped_data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the reshaped target data.

    Parameters
    ----------
    reshaped_data : pl.LazyFrame
        LazyFrame representing the reshaped target data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the cleaned target data.

    Notes
    -----
    This function cleans the reshaped target data to ensure consistency
    and readability.
    """
    col_mapper = {"Online or In-Person": "transaction_method"}

    # Expressions
    quarter_expr = pl.col("quarter").str.extract(r"(\d)").cast(pl.Int8)

    return reshaped_data.with_columns(quarter_expr).rename(col_mapper)


def view_dsb_variance_to_target(
    pre_transaction: pl.LazyFrame, pre_target: pl.LazyFrame
) -> pl.LazyFrame:
    """View the variance to target for DSB transactions.

    Parameters
    ----------
    pre_transaction : pl.LazyFrame
        LazyFrame representing the preprocessed transaction data.
    pre_target : pl.LazyFrame
        LazyFrame representing the preprocessed target data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing transaction data with variance to target calculated.

    Notes
    -----
    This function calculates the variance to target for DSB transactions
    by joining preprocessed transaction and target dataframes and computing
    the difference between the total transaction value and the target value
    for each quarter and transaction method.
    """
    # Aggregate the total value of DSB transactions
    dsb_total_value_by_quarter_method = pre_transaction.pipe(
        aggregate_total_dsb_value_by_quarter_method
    )

    # Expressions
    variance_to_target_expr = pl.col("total_value") - pl.col("target")

    return dsb_total_value_by_quarter_method.join(
        pre_target, on=["transaction_method", "quarter"]
    ).with_columns(variance_to_target=variance_to_target_expr)


def aggregate_total_dsb_value_by_quarter_method(
    pre_transaction: pl.LazyFrame,
) -> pl.LazyFrame:
    """Aggregate total transaction value for DSB bank by calendar quarter
    and transaction method.

    Parameters
    ----------
    pre_transaction : pl.LazyFrame
        LazyFrame representing the preprocessed transaction data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the aggregated data, with total transaction
        value grouped by calendar quarter and transaction method.

    Notes
    -----
    This function filters the preprocessed transaction data for transactions
    with DSB bank, aggregates the total transaction value for each calendar
    quarter and transaction method, and returns the result.
    """
    quarter_expr = pl.col("created_on").dt.quarter()

    return (
        pre_transaction.filter(pl.col("bank") == "DSB")
        .with_columns(quarter=quarter_expr)
        .pipe(
            aggregate_total_value,
            "transaction_method",
            "quarter",
        )
    )
