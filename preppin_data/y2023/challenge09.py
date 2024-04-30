"""2023: Week 9 - Customer Bank Statements

Inputs
------
- __input/2023/Account Information.csv
- __input/2023/Transaction Detail.csv
- __input/2023/Transaction Path.csv

Outputs
-------
- output/2023/wk09_customer_statement.ndjson

Notes
-----
We've modified the suggested return, so we only return transactions for
the current month (i.e., we filter out the January transactions.)
"""

import polars as pl

from .challenge07 import (
    load_data,
    preprocess_account_info,
    preprocess_transaction_detail,
    preprocess_transaction_path,
)


def solve(
    account_info_fsrc: str,
    transaction_detail_fsrc: str,
    transaction_path_fsrc: str,
) -> pl.DataFrame:
    """Solve challenge 9 of Preppin' Data 2023.

    Parameters
    ----------
    account_info_fsrc : str
        Filepath of the input CSV file containing account information data.
    transaction_detail_fsrc : str
        Filepath of the input CSV file containing transaction detail data.
    transaction_path_fsrc : str
        Filepath of the input CSV file containing transaction path data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the customer statements for the current month.

    Notes
    -----
    This function loads account information, transaction detail, and transaction
    path data  from CSV files. It preprocesses the data and generates customer
    statements for the  current month, including running balances for each
    account.
    """

    # Load the data
    account_info = load_data(account_info_fsrc)

    transaction_detail = load_data(transaction_detail_fsrc)

    transaction_path = load_data(transaction_path_fsrc)

    # Preprocess the data
    pre_account_info = account_info.pipe(preprocess_account_info)

    pre_transaction_detail = transaction_detail.pipe(preprocess_transaction_detail)

    pre_transaction_path = transaction_path.pipe(preprocess_transaction_path)

    return view_customer_statement(
        pre_account_info,
        pre_transaction_detail,
        pre_transaction_path,
    ).collect()


def view_customer_statement(
    pre_account_info: pl.LazyFrame,
    pre_transaction_detail: pl.LazyFrame,
    pre_transaction_path: pl.LazyFrame,
) -> pl.LazyFrame:
    """View the customer statement for the current month.

    Parameters
    ----------
    pre_account_info : pl.LazyFrame
        LazyFrame representing the preprocessed account information data.
    pre_transaction_detail : pl.LazyFrame
        LazyFrame representing the preprocessed transaction detail data.
    pre_transaction_path : pl.LazyFrame
        LazyFrame representing the preprocessed transaction path data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing the customer statements for the current month.

    Notes
    -----
    This function generates customer statements for the current month based
    on preprocessed account information, transaction detail, and transaction
    path data. It calculates running  balances for each account and filters
    the statements for the current month.
    """
    # Scaffold the data
    customer_statement = scaffold_customer_statement(
        pre_account_info,
        pre_transaction_detail,
        pre_transaction_path,
    )

    # Predicates
    current_month_pred = pl.col("transaction_created_on").dt.month() == 2

    # Expressions
    running_balance_expr = pl.cum_sum("transaction_value").over("account_number")

    return (
        customer_statement.sort(
            "account_number",
            "transaction_created_on",
            "transaction_value",
            descending=[False, False, True],
        )
        .with_columns(running_balance=running_balance_expr)
        .filter(current_month_pred)
    )


def scaffold_customer_statement(
    pre_account_info: pl.LazyFrame,
    pre_transaction_detail: pl.LazyFrame,
    pre_transaction_path: pl.LazyFrame,
) -> pl.LazyFrame:
    """Scaffold the customer statement.

    Parameters
    ----------
    pre_account_info : pl.LazyFrame
        LazyFrame representing the preprocessed account information data.
    pre_transaction_detail : pl.LazyFrame
        LazyFrame representing the preprocessed transaction detail data.
    pre_transaction_path : pl.LazyFrame
        LazyFrame representing the preprocessed transaction path data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing the scaffolded customer statements.

    Notes
    -----
    This function scaffolds the customer statements by combining initial
    balance data from preprocessed account information and daily transaction
    flow data from preprocessed transaction detail and transaction path
    data. It generates a diagonal concatenation of  initial balances and
    daily transaction flows for each account.
    """

    # Collect the data
    daily_transaction_flow = view_transaction_flow(
        pre_transaction_detail, pre_transaction_path
    )

    initial_balance = pre_account_info.unique(["account_number"]).select(
        "account_number",
        transaction_created_on="balance_taken_on",
        transaction_value="balance",
    )

    return pl.concat([initial_balance, daily_transaction_flow], how="diagonal")


def view_transaction_flow(
    pre_transaction_detail: pl.LazyFrame,
    pre_transaction_path: pl.LazyFrame,
) -> pl.LazyFrame:
    """View the transaction flow.

    Parameters
    ----------
    pre_transaction_detail : pl.LazyFrame
        LazyFrame representing the preprocessed transaction detail data.
    pre_transaction_path : pl.LazyFrame
        LazyFrame representing the preprocessed transaction path data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing the transaction flow data.

    Notes
    -----
    This function views the transaction flow by joining preprocessed transaction
    detail and transaction path data. It filters out cancelled transactions
    and calculates the signed transaction values based on source and destination
    variables. The resulting LazyFrame represents the transaction flow for
    further analysis.
    """

    # Predicates
    is_not_cancelled_pred = pl.col("is_cancelled").not_()

    # Expressions
    sign_transaction_expr = (
        pl.when(pl.col("variable").str.starts_with("destination"))
        .then(pl.lit(1))
        .when(pl.col("variable").str.starts_with("source"))
        .then(pl.lit(-1))
        .otherwise(pl.lit(None))
    )

    signed_transaction_value = pl.col("transaction_value") * sign_transaction_expr

    return (
        pre_transaction_path.melt(id_vars="transaction_id", value_name="account_number")
        .join(pre_transaction_detail, on="transaction_id")
        .filter(is_not_cancelled_pred)
        .select(
            "account_number",
            signed_transaction_value,
            "transaction_created_on",
        )
    )
