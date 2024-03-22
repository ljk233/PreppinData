"""2023.challenge07.py
"""

import polars as pl


def solve(
    account_holder_fsrc: str,
    account_info_fsrc: str,
    transaction_detail_fsrc: str,
    transaction_path_fsrc: str,
    output_ndjson_fdst: str,
) -> pl.LazyFrame:
    # Preprocess the data
    pre_account_holder = preprocess_account_holder(account_holder_fsrc)
    pre_account_info = preprocess_account_info(account_info_fsrc)
    pre_transaction_detail = preprocess_transaction_detail(transaction_detail_fsrc)
    pre_transaction_path = preprocess_transaction_path(transaction_path_fsrc)

    # Select non-platinum accounts
    filtered_account_info = pre_account_info.filter(
        pl.col("account_type") != "Platinum"
    )

    # Select transactions that were not cancelled and had a value greater than 1000
    filtered_transaction_detail = pre_transaction_detail.filter(
        (pl.col("is_cancelled").not_()) & (pl.col("transaction_value") > 1000)
    )

    # Merge the datasets
    flagged_transactions = (
        filtered_transaction_detail.join(pre_transaction_path, on="transaction_id")
        .join(
            filtered_account_info,
            left_on="outbound_account_number",
            right_on="account_number",
        )
        .join(pre_account_holder, left_on="account_holder_id", right_on="id")
        .select(
            "transaction_id",
            "inbound_account_number",
            "created_on",
            "transaction_value",
            "is_cancelled",
            "outbound_account_number",
            "balance_taken_on",
            "balance",
            "full_name",
            "date_of_birth",
            "mobile_contact",
            "first_line_of_address",
        )
    )

    # Export the data
    flagged_transactions.collect().write_ndjson(output_ndjson_fdst)

    return flagged_transactions


def preprocess_account_holder(fsrc: str) -> pl.LazyFrame:
    """Preprocess the account holder data."""
    return (
        pl.scan_csv(fsrc)
        .with_columns(
            pl.col("Date of Birth").str.to_date(),
            pl.col("Contact Number").cast(pl.Utf8).str.pad_start(11, "0"),
        )
        .rename(
            {
                "Account Holder ID": "id",
                "Name": "full_name",
                "Date of Birth": "date_of_birth",
                "Contact Number": "mobile_contact",
                "First Line of Address": "first_line_of_address",
            }
        )
    )


def preprocess_account_info(fsrc: str) -> pl.LazyFrame:
    """Preprocess the account information data."""
    return (
        pl.scan_csv(fsrc)
        .with_columns(
            pl.col("Account Holder ID").str.split(", "),
            pl.col("Balance Date").str.to_date(),
        )
        .explode("Account Holder ID")
        .with_columns(pl.col("Account Holder ID").cast(pl.Int64))
        .rename(
            {
                "Account Number": "account_number",
                "Account Type": "account_type",
                "Account Holder ID": "account_holder_id",
                "Balance Date": "balance_taken_on",
                "Balance": "balance",
            }
        )
    )


def preprocess_transaction_detail(fsrc: str) -> pl.LazyFrame:
    """Preprocess the transaction detail data."""
    return (
        pl.scan_csv(fsrc)
        .with_columns(
            pl.col("Transaction Date").str.to_date().alias("created_on"),
            (
                pl.when(pl.col("Cancelled?") == "Y")
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias("is_cancelled")
            ),
        )
        .rename({"Transaction ID": "transaction_id", "Value": "transaction_value"})
        .select("created_on", "transaction_id", "transaction_value", "is_cancelled")
    )


def preprocess_transaction_path(fsrc: str) -> pl.LazyFrame:
    """Preprocess the transaction path data."""
    return pl.scan_csv(fsrc).rename(
        {
            "Transaction ID": "transaction_id",
            "Account_From": "outbound_account_number",
            "Account_To": "inbound_account_number",
        }
    )
