"""2023: Week 10 - What's my Balance on this day?

Inputs
------
- __input/2023/Account Information.csv
- __input/2023/Transaction Detail.csv
- __input/2023/Transaction Path.csv

Outputs
-------
- output/2023/wk10_daily_account_summary_01FEB.ndjson
"""

import polars as pl

from . import challenge09


def solve(
    account_info_fsrc: str,
    transaction_detail_fsrc: str,
    transaction_path_fsrc: str,
    summary_on: str = "2023-02-01",
) -> pl.LazyFrame:
    """Solves challenge 10 of Preppin' Data 2023.

    Parameters
    ----------
    account_info_fsrc : str
        Filepath of the input CSV file containing account information data.
    transaction_detail_fsrc : str
        Filepath of the input CSV file containing transaction detail data.
    transaction_path_fsrc : str
        Filepath of the input CSV file containing transaction path data.
    summary_on : str, optional
        Date to return the account summary for. Must be in format 'YYYY-MM-DD'.
        Default is '2023-02-01'.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the account summary for the specified date.

    Notes
    -----
    This function solves the challenge by processing account and transaction
    data to generate an account summary for a given date.
    """

    # Load the data
    running_balance = challenge09.solve(
        account_info_fsrc,
        transaction_detail_fsrc,
        transaction_path_fsrc,
    )

    account_info = (
        challenge09.load_data(account_info_fsrc)
        .pipe(challenge09.preprocess_account_info)
        .collect()
    )

    return view_account_summary_on(
        running_balance,
        account_info,
        summary_on,
    )


def view_account_summary_on(
    running_balance: pl.DataFrame,
    account_info: pl.DataFrame,
    summary_on: str,
) -> pl.DataFrame:
    """Returns the account summary for all account numbers on the given
    sumary_on date.

    Parameters
    ----------
    running_balance : pl.DataFrame
        DataFrame containing running balance information.
    account_info : pl.DataFrame
        DataFrame containing account information.
    summary_on : str
        Date for which to retrieve the account summary. Must be in format
        'YYYY-MM-DD'.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the account summary for the specified date.

    Notes
    -----
    This function retrieves the account summary for all account numbers
    on the specified date by filtering the running balance and account
    information DataFrames accordingly.
    """

    # Collect the data
    daily_account_summary = view_daily_account_summary(running_balance, account_info)

    # Expressions
    summary_on_expr = pl.lit(summary_on).str.to_date()

    # Predicates
    summary_on_pred = pl.col("summary_on") == summary_on_expr

    return daily_account_summary.filter(summary_on_pred)


def view_daily_account_summary(
    running_balance: pl.DataFrame, account_info: pl.DataFrame
) -> pl.DataFrame:
    """Returns the customer total transactions and daily balance.

    Parameters
    ----------
    running_balance : pl.DataFrame
        DataFrame containing running balance information.
    account_info : pl.DataFrame
        DataFrame containing account information.

    Returns
    -------
    pl.DataFrame
        DataFrame containing customer total transactions and daily balance.

    Notes
    -----
    This function calculates the total transactions and daily balance for
    each customer based on the provided running balance and account information
    DataFrames.
    """

    # Collect the data
    scaffolded_data = scaffold_data(running_balance, account_info)

    total_daily_transactions = running_balance.pipe(view_total_daily_transactions)

    daily_end_balance = running_balance.pipe(view_daily_end_balance)

    # Expressions
    fill_initial_daily_end_balance_expr = (
        pl.when(pl.col("summary_on") == pl.min("summary_on"))
        .then("month_end_balance")
        .otherwise("daily_end_balance")
    )

    fill_null_total_transaction_value_expr = pl.col(
        "total_transaction_value"
    ).fill_null(0.0)

    fill_null_daily_end_balance_expr = (
        pl.col("daily_end_balance").forward_fill().over("account_number")
    )

    # Predicates
    filter_out_min_summary_on_expr = pl.col("summary_on") != pl.min("summary_on")

    return (
        scaffolded_data.join(
            total_daily_transactions,
            left_on=["account_number", "summary_on"],
            right_on=["account_number", "transaction_created_on"],
            how="left",
        )
        .join(
            daily_end_balance,
            left_on=["account_number", "summary_on"],
            right_on=["account_number", "balance_taken_on"],
            how="left",
        )
        .with_columns(daily_end_balance=fill_initial_daily_end_balance_expr)
        .with_columns(
            fill_null_total_transaction_value_expr,
            fill_null_daily_end_balance_expr,
        )
        .filter(filter_out_min_summary_on_expr)
        .drop("month_end_balance")
        .sort("account_number", "summary_on")
    )


def scaffold_data(
    running_balance: pl.DataFrame, account_info: pl.DataFrame
) -> pl.DataFrame:
    """Scaffolds the data so each account number has a record for all dates
    of interest.

    Parameters
    ----------
    running_balance : pl.DataFrame
        DataFrame containing running balance information.
    account_info : pl.DataFrame
        DataFrame containing account information.

    Returns
    -------
    pl.DataFrame
        Scaffolded DataFrame with records for all account numbers and dates
        of interest.

    Notes
    -----
    This function generates a scaffolded DataFrame with records for all
    account numbers and dates of interest, ensuring each account number
    has a record for every date.
    """

    # Collect the data
    prev_month_end_balance = account_info.pipe(view_previous_month_end_balance)

    account_number = account_info.pipe(view_unique_account_number)

    full_sumary_on = running_balance.pipe(view_full_summary_on)

    this_month_account_summary_on = account_number.join(full_sumary_on, how="cross")

    return pl.concat(
        [prev_month_end_balance, this_month_account_summary_on], how="diagonal"
    )


def view_previous_month_end_balance(account_info: pl.DataFrame) -> pl.DataFrame:
    """Returns the account numbers' previous month end balance.

    Parameters
    ----------
    account_info : pl.DataFrame
        DataFrame containing account information.

    Returns
    -------
    pl.DataFrame
        DataFrame with account numbers and their previous month end balance.

    Notes
    -----
    This function retrieves the previous month end balance for each account
    number based on the provided account information DataFrame.
    """

    return account_info.select(
        "account_number",
        month_end_balance="balance",
        summary_on="balance_taken_on",
    ).unique()


def view_unique_account_number(account_info: pl.DataFrame) -> pl.DataFrame:
    """Returns the unique collection of account numbers.

    Parameters
    ----------
    account_info : pl.DataFrame
        DataFrame containing account information.

    Returns
    -------
    pl.DataFrame
        DataFrame with unique account numbers.

    Notes
    -----
    This function retrieves the unique collection of account numbers from
    the provided account information DataFrame.
    """

    return account_info.select("account_number").unique()


def view_full_summary_on(running_balance: pl.DataFrame) -> pl.DataFrame:
    """Returns the unique collection of transaction creation dates.

    Parameters
    ----------
    running_balance : pl.DataFrame
        DataFrame containing running balance information.

    Returns
    -------
    pl.DataFrame
        DataFrame with unique transaction creation dates.

    Notes
    -----
    This function retrieves the unique collection of transaction creation
    dates from the provided running balance DataFrame.
    """

    return (
        running_balance.select(summary_on="transaction_created_on")
        .unique()
        .sort("summary_on")
        .upsample("summary_on", every="1d")
    )


def view_total_daily_transactions(running_balance: pl.DataFrame) -> pl.DataFrame:
    """Aggregates the transaction value by account number and transaction
    creation date.

    Parameters
    ----------
    running_balance : pl.DataFrame
        DataFrame containing running balance information.

    Returns
    -------
    pl.DataFrame
        DataFrame with total transaction value aggregated by account number
        and transaction creation date.

    Notes
    -----
    This function aggregates the transaction value by account number and
    transaction creation date from the provided running balance DataFrame.
    """

    return running_balance.group_by("account_number", "transaction_created_on").agg(
        total_transaction_value=pl.sum("transaction_value")
    )


def view_daily_end_balance(running_balance: pl.DataFrame) -> pl.DataFrame:
    """Returns the daily end balance of account numbers.

    Parameters
    ----------
    running_balance : pl.DataFrame
        DataFrame containing running balance information.

    Returns
    -------
    pl.DataFrame
        DataFrame with daily end balance of account numbers.

    Notes
    -----
    This function calculates the daily end balance of account numbers based
    on the provided running balance DataFrame.
    """

    return running_balance.group_by(
        "account_number", balance_taken_on="transaction_created_on"
    ).agg(daily_end_balance=pl.last("running_balance"))
