"""2023: Week 1 - The Data Source Bank

Inputs
------
- __input/2023/PD 2023 Wk 1 Input.csv

Outputs
-------
- output/2023/wk01_total_value_by_bank.ndjson
- output/2023/wk01_total_value_by_bank_method_weekday.ndjson
- output/2023/wk01_total_value_by_bank_customer_code.ndjson
"""

import polars as pl


def solve(pd_input_wk1_fsrc: str) -> tuple[pl.DataFrame]:
    """Solve challenge 1 of Preppin' Data 2023.

    Parameters
    ----------
    pd_input_wk1_fsrc : str
        Filepath of the input CSV file for Week 1.

    Returns
    -------
    tuple[pl.DataFrame]
        A tuple containing three DataFrames representing the aggregated data:
        1. Total transaction value grouped by bank.
        2. Total transaction value grouped by bank, transaction method,
           and weekday.
        3. Total transaction value grouped by bank and customer code.

    Notes
    -----
    This function reads input data from a CSV file, preprocesses it, and
    then aggregates the total transaction values by different criteria.
    """
    # Load the data
    transaction = load_transactions_data(pd_input_wk1_fsrc)

    # Preprocess the data
    pre_transaction = transaction.pipe(preprocess_transactions_data)

    # Aggregate the data
    return (
        pre_transaction.pipe(aggregate_total_value_by_bank).collect(),
        pre_transaction.pipe(aggregate_total_value_by_bank_method_weekday).collect(),
        pre_transaction.pipe(aggregate_total_value_by_bank_customer_code).collect(),
    )


def load_transactions_data(pd_input_wk1_fsrc: str) -> pl.LazyFrame:
    """Load data from a CSV file.

    Parameters
    ----------
    pd_input_wk1_fsrc : str
        Filepath of the input CSV file for Week 1.

    Returns
    -------
    pl.DataFrame
        LazyFrame representing the data.
    """
    data = pl.scan_csv(pd_input_wk1_fsrc, try_parse_dates=True)

    return data


def preprocess_transactions_data(data: pl.DataFrame) -> pl.LazyFrame:
    """Preprocess the source data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the source data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed source data.

    Notes
    -----
    Primary key is {transaction_code}.
    """
    return data.pipe(clean_transactions_data)


def clean_transactions_data(data: pl.DataFrame) -> pl.LazyFrame:
    """Clean the source data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the source data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the clean source data.
    """
    col_mapper = {
        "Transaction Code": "transaction_code",
        "Value": "value",
        "Customer Code": "customer_code",
        "Transaction Date": "created_on",
    }

    # Expressions
    bank_expr = pl.col("Transaction Code").str.extract(r"^(\w+)")

    created_on_expr = pl.col("Transaction Date").cast(pl.Date)

    transaction_method_expr = (
        pl.when(pl.col("Online or In-Person") == 1)
        .then(pl.lit("Online"))
        .when(pl.col("Online or In-Person") == 2)
        .then(pl.lit("In-Person"))
        .otherwise(pl.lit(None))
    )

    return (
        data.with_columns(
            created_on_expr,
            transaction_method=transaction_method_expr,
            bank=bank_expr,
        )
        .drop("Online or In-Person")
        .rename(col_mapper)
    )


def aggregate_total_value_by_bank(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate total transaction value by bank.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing the preprocessed source data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the aggregated data, with total transaction value
        grouped by bank.
    """
    return pre_data.pipe(aggregate_total_value, "bank")


def aggregate_total_value_by_bank_method_weekday(
    pre_data: pl.LazyFrame,
) -> pl.LazyFrame:
    """Aggregate total transaction value by bank, transaction method, and weekday.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing the preprocessed source data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the aggregated data, with total transaction value
        grouped by bank, transaction method, and weekday.
    """
    weekday_expr = pl.col("created_on").dt.strftime("%A")

    return pre_data.with_columns(weekday=weekday_expr).pipe(
        aggregate_total_value,
        "bank",
        "transaction_method",
        "weekday",
    )


def aggregate_total_value_by_bank_customer_code(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate total transaction value by bank and customer code.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing the preprocessed source data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the aggregated data, with total transaction value
        grouped by bank and customer code.
    """
    return pre_data.pipe(aggregate_total_value, "bank", "customer_code")


def aggregate_total_value(pre_data: pl.LazyFrame, *group_by: str) -> pl.LazyFrame:
    """Aggregate total transaction value by specified groupings.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing the preprocessed source data.
    *group_by : str
        Columns to group the data by.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the aggregated data, with total transaction value
        grouped by the specified columns.
    """
    return pre_data.group_by(group_by).agg(total_value=pl.sum("value"))
