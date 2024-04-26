"""2023: Week 2 - International Bank Account Numbers

Inputs
------
- __input/2023/Transactions.csv
- __input/2023/Swift Codes.csv

Outputs
-------
- output/2023/wk02_transaction_iban.ndjson
"""

import polars as pl


def solve(transactions_fsrc: str, swift_fsrc: str) -> pl.DataFrame:
    """Solve challenge 1 of Preppin' Data 2023.

    Parameters
    ----------
    transactions_fsrc : str
        Filepath of the input CSV file containing transaction data.
    swift_fsrc : str
        Filepath of the input CSV file containing SWIFT data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing transaction IDs and corresponding IBANs.

    Notes
    -----
    This function reads transaction data and SWIFT data from CSV files,
    preprocesses them, and generates International Bank Account Numbers
    (IBANs) for the transactions based on the provided SWIFT information.
    """
    # Load and preprocess the data
    pre_transaction = load_data(transactions_fsrc).pipe(preprocess_transaction_data)

    pre_swift = load_data(swift_fsrc).pipe(preprocess_swift_data)

    return view_transaction_iban(pre_transaction, pre_swift).collect()


def load_data(fsrc: str) -> pl.LazyFrame:
    """Load transaction data from a CSV file.

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


def preprocess_transaction_data(data: pl.DataFrame) -> pl.LazyFrame:
    """Preprocess the transactions data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the transactions data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed transactions data.

    Notes
    -----
    Primary key is {transaction_id}.
    """
    return data.pipe(clean_transaction_data)


def clean_transaction_data(data: pl.DataFrame) -> pl.LazyFrame:
    """Clean the transactions data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the transactions data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the clean transactions data.

    Notes
    -----
    This function performs the following cleaning steps:
    - Renames columns for consistency and readability.
    - Removes hyphens from the sort code and casts it to integer type.
    """
    col_mapper = {
        "Transaction ID": "transaction_id",
        "Account Number": "account_number",
        "Sort Code": "sort_code",
        "Bank": "bank_name",
    }

    # Expressions
    sort_code_expr = pl.col("Sort Code").str.replace_all("-", "").cast(pl.Int64)

    return data.with_columns(sort_code_expr).rename(col_mapper)


def preprocess_swift_data(data: pl.DataFrame) -> pl.LazyFrame:
    """Preprocess the swift data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the swift data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed swift data.

    Notes
    -----
    There are two candidates for the primary key: {bank_name | swift_code}.
    """
    return data.pipe(clean_swift_data)


def clean_swift_data(data: pl.DataFrame) -> pl.LazyFrame:
    """Clean the swift data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the swift data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the clean swift data.

    Notes
    -----
    This function performs the following cleaning steps:
    - Renames columns for consistency and readability.
    """
    col_mapper = {
        "Bank": "bank_name",
        "SWIFT code": "swift_code",
        "Check Digits": "check_digits",
    }

    return data.rename(col_mapper)


def view_transaction_iban(
    pre_transaction: pl.LazyFrame, pre_swift: pl.LazyFrame
) -> pl.LazyFrame:
    """Generate International Bank Account Numbers (IBANs) for transactions.

    Parameters
    ----------
    pre_transaction : pl.LazyFrame
        LazyFrame representing the preprocessed transaction data.
    pre_swift : pl.LazyFrame
        LazyFrame representing the preprocessed SWIFT data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing transaction IDs and corresponding IBANs.

    Notes
    -----
    This function performs the following operations:
    - Constructs IBANs using the provided transaction and SWIFT data.
    - Joins transaction data with SWIFT data on bank name.
    - Creates IBANs based on a combination of SWIFT code, sort code, and
      account number.
    """
    # Expressions
    iban_expr = (
        pl.lit("GB")
        + pl.col("check_digits")
        + pl.col("swift_code")
        + pl.col("sort_code").cast(pl.Utf8)
        + pl.col("account_number").cast(pl.Utf8)
    )

    return pre_transaction.join(pre_swift, on="bank_name").select(
        "transaction_id", iban=iban_expr
    )
