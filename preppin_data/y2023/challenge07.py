"""2023: Week 7 - Flagging Fraudulent Suspicions

Inputs
------
- __input/2023/Account Holders.csv
- __input/2023/Account Information.csv
- __input/2023/Transaction Detail.csv
- __input/2023/Transaction Path.csv

Outputs
-------
- output/2023/wk07_flagged_transactions.ndjson
"""

import polars as pl


def solve(
    account_holder_fsrc: str,
    account_info_fsrc: str,
    transaction_detail_fsrc: str,
    transaction_path_fsrc: str,
) -> pl.DataFrame:
    """Solve challenge 7 of Preppin' Data 2023.

    Parameters
    ----------
    account_holder_fsrc : str
        Filepath of the input CSV file containing account holder data.

    account_info_fsrc : str
        Filepath of the input CSV file containing account information data.

    transaction_detail_fsrc : str
        Filepath of the input CSV file containing transaction detail data.

    transaction_path_fsrc : str
        Filepath of the input CSV file containing transaction path data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the flagged transactions for further investigation.

    Notes
    -----
    This function loads, preprocesses, and analyzes multiple datasets to identify
    potentially fraudulent transactions. It involves loading account holder data,
    account information data, transaction detail data, and transaction path
    data. Each dataset undergoes preprocessing to clean and reshape the
    data as necessary. The function then examines the transactions, applying
    specific criteria to flag suspicious transactions for further investigation.
    Finally, it returns a DataFrame containing the flagged transactions.
    """

    # Load the data
    account_holder = load_data(account_holder_fsrc)

    account_info = load_data(account_info_fsrc)

    transaction_detail = load_data(transaction_detail_fsrc)

    transaction_path = load_data(transaction_path_fsrc)

    # Preprocess the data
    pre_account_holder = account_holder.pipe(preprocess_account_holder)

    pre_account_info = account_info.pipe(preprocess_account_info)

    pre_transaction_detail = transaction_detail.pipe(preprocess_transaction_detail)

    pre_transaction_path = transaction_path.pipe(preprocess_transaction_path)

    return view_flagged_transactions(
        pre_account_holder,
        pre_account_info,
        pre_transaction_detail,
        pre_transaction_path,
    ).collect()


def load_data(fsrc: str) -> pl.LazyFrame:
    """Load the data from the given CSV file.

    Parameters
    ----------
    fsrc: str
        File path to the input CSV file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the data from the input CSV file.
    """

    return pl.scan_csv(fsrc, try_parse_dates=True)


def preprocess_account_holder(data: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocess the account holder data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the account holder data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed account holder data.

    Notes
    -----
    Primary key is {account_holder_id}.
    """

    return data.pipe(clean_account_holder)


def clean_account_holder(data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the account holder data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the account holder data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the cleaned account holder data.
    """

    col_mapper = {
        "Account Holder ID": "account_holder_id",
        "Name": "full_name",
        "Date of Birth": "date_of_birth",
        "Contact Number": "mobile_contact",
        "First Line of Address": "first_line_of_address",
    }

    # Expressions
    contact_number_expr = pl.col("Contact Number").cast(pl.Utf8).str.pad_start(11, "0")

    return data.with_columns(contact_number_expr).rename(col_mapper)


def preprocess_account_info(data: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocess the account information data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the account information data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed account information data.

    Notes
    -----
    Primary key is {account_number, account_created_on}.
    """

    return data.pipe(reshape_account_info).pipe(clean_account_info)


def reshape_account_info(reshaped_data: pl.LazyFrame) -> pl.LazyFrame:
    """Reshape the account information data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the account information data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the reshaped account information data.
    """

    # Expressions
    split_account_holder_id_expr = pl.col("Account Holder ID").str.split(", ")

    parse_account_holder_id_expr = pl.col("Account Holder ID").cast(pl.Int64)

    return (
        reshaped_data.with_columns(split_account_holder_id_expr)
        .explode("Account Holder ID")
        .with_columns(parse_account_holder_id_expr)
    )


def clean_account_info(reshaped_data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the account information data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the account information data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the cleaned account information data.
    """

    col_mapper = {
        "Account Number": "account_number",
        "Account Type": "account_type",
        "Account Holder ID": "account_holder_id",
        "Balance Date": "balance_taken_on",
        "Balance": "balance",
    }

    return reshaped_data.rename(col_mapper)


def preprocess_transaction_detail(data: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocess the transaction detail data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the transaction detail data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed transaction detail data.

    Notes
    -----
    Primary key is {transaction_id}.
    """

    return data.pipe(clean_transaction_detail)


def clean_transaction_detail(data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the transaction detail data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the transaction detail data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the cleaned transaction detail data.
    """

    col_mapper = {
        "Transaction ID": "transaction_id",
        "Transaction Date": "transaction_created_on",
        "Value": "transaction_value",
    }

    # Expressions
    is_cancelled_expr = (
        pl.when(pl.col("Cancelled?") == "Y")
        .then(pl.lit(True))
        .when(pl.col("Cancelled?") == "N")
        .then(pl.lit(False))
        .otherwise(pl.lit(None))
    )

    return (
        data.with_columns(is_cancelled=is_cancelled_expr)
        .rename(col_mapper)
        .drop("Cancelled?")
    )


def preprocess_transaction_path(data: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocess the transaction path data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the transaction path data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed transaction path data.

    Notes
    -----
    Primary key is {transaction_id}.
    """

    return data.pipe(clean_transaction_path)


def clean_transaction_path(data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the transaction path data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the transaction path data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the cleaned transaction path data.
    """

    col_mapper = {
        "Transaction ID": "transaction_id",
        "Account_From": "source_account_number",
        "Account_To": "destination_account_number",
    }

    return data.rename(col_mapper)


def view_flagged_transactions(
    pre_account_holder: pl.LazyFrame,
    pre_account_info: pl.LazyFrame,
    pre_transaction_detail: pl.LazyFrame,
    pre_transaction_path: pl.LazyFrame,
) -> pl.LazyFrame:
    """View candidate fraudulent transactions.

    Parameters
    ----------
    pre_account_holder : pl.LazyFrame
        LazyFrame representing the preprocessed account holder data.

    pre_account_info : pl.LazyFrame
        LazyFrame representing the preprocessed account information data.

    pre_transaction_detail : pl.LazyFrame
        LazyFrame representing the preprocessed transaction detail data.

    pre_transaction_path : pl.LazyFrame
        LazyFrame representing the preprocessed transaction path data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the flagged transactions for further investigation.

    Notes
    -----
    This function identifies potential fraudulent transactions based on specified
    criteria. It joins the preprocessed dataframes representing account holders,
    account information, transaction details, and transaction paths, applies
    filtering conditions to isolate transactions of interest, and selectsrelevant columns for further analysis or reporting.
    """

    # Predicates
    is_platinum_account_expr = pl.col("account_type") != "Platinum"

    is_not_cancelled_expr = pl.col("is_cancelled").not_()

    is_greater_1000_expr = pl.col("transaction_value") > 1000

    return (
        pre_transaction_detail.join(pre_transaction_path, on="transaction_id")
        .join(
            pre_account_info,
            left_on="source_account_number",
            right_on="account_number",
        )
        .join(pre_account_holder, on="account_holder_id")
        .filter(is_platinum_account_expr & is_not_cancelled_expr & is_greater_1000_expr)
        .select(
            "transaction_id",
            "destination_account_number",
            "transaction_created_on",
            "transaction_value",
            "is_cancelled",
            "source_account_number",
            "balance_taken_on",
            "balance",
            "full_name",
            "date_of_birth",
            "mobile_contact",
            "first_line_of_address",
        )
    )
