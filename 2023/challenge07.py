"""2023.challenge07.py
"""

import polars as pl


# Parameters
# ----------

# Input
# =====
ACCOUNT_HOLDERS_CSV = "2023/data/input/Account Holders.csv"
ACCOUNT_INFO_CSV = "2023/data/input/Account Information.csv"
TRANSACTION_DETAIL_CSV = "2023/data/input/Transaction Detail.csv"
TRANSACTION_PATH_CSV = "2023/data/input/Transaction Path.csv"

# STAGE
# =====
STG_ACCOUNT_HOLDERS_PQT = "2023/data/stage/account_holders.parquet"
STG_ACCOUNT_INFO_PQT = "2023/data/stage/account_info.parquet"
STG_TRANSACTION_DETAIL_PQT = "2023/data/stage/transaction_detail.parquet"
STG_TRANSACTION_PATH_PQT = "2023/data/stage/transaction_path.parquet"

# Output
# ======
FLAGGED_TRANSACTIONS_NDJSON = "2023/data/output/flagged_transction.ndjson"

# Functions
# ---------


def main():
    """Main function for identifying potentially fraudalent transactions."""
    # Preprocess the data
    account_holder = preprocess_account_holder(ACCOUNT_HOLDERS_CSV)
    account_info = preprocess_account_info(ACCOUNT_INFO_CSV)
    transaction_detail = preprocess_transaction_detail(TRANSACTION_DETAIL_CSV)
    transaction_path = preprocess_transaction_path(TRANSACTION_PATH_CSV)

    # Export the preprocessed data
    export_preprocessed_data(
        (account_holder, STG_ACCOUNT_HOLDERS_PQT),
        (account_info, STG_ACCOUNT_INFO_PQT),
        (transaction_detail, STG_TRANSACTION_DETAIL_PQT),
        (transaction_path, STG_TRANSACTION_PATH_PQT),
    )

    # Select non-platinum accounts
    selected_account_info = account_info.pipe(filter_account_info)

    # Select transactions that were not cancelled and had a value greater than 1000
    selected_transaction_detail = transaction_detail.pipe(filter_transaction_detail)

    # Merge the datasets
    flagged_transactions = merge_data(
        account_holder,
        selected_account_info,
        selected_transaction_detail,
        transaction_path,
    )

    # Export the data
    flagged_transactions.collect().write_ndjson(FLAGGED_TRANSACTIONS_NDJSON)


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
                "First Line of Address": "home_address_1",
            }
        )
    )


def preprocess_account_info(fsrc: str) -> pl.LazyFrame:
    """Preprocess the account information data."""
    return (
        pl.scan_csv(fsrc)
        .filter(pl.col("Account Type") != "Platinum")
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
            "Account_From": "source_account_number",
            "Account_To": "target_account_number",
        }
    )


def export_preprocessed_data(*args: tuple[pl.LazyFrame, str]) -> None:
    """Export preprocessed data to Parquet format."""
    for data, fdst in args:
        data.collect().write_parquet(fdst)


def filter_account_info(account_holder: pl.LazyFrame) -> pl.LazyFrame:
    """Selected non-Platinum account types."""
    return account_holder.filter(pl.col("account_type") != "Platinum")


def filter_transaction_detail(transaction_detail: pl.LazyFrame) -> pl.LazyFrame:
    """Select transaction that are not cancelled and have a value greater
    than 1000.
    """
    return transaction_detail.filter(
        (pl.col("is_cancelled").not_()) & (pl.col("transaction_value") > 1000)
    )


def merge_data(
    account_holder: pl.LazyFrame,
    account_info: pl.LazyFrame,
    transaction_detail: pl.LazyFrame,
    transaction_path: pl.LazyFrame,
) -> pl.LazyFrame:
    """Merge the data into the final dataset."""
    return (
        transaction_detail.join(transaction_path, on="transaction_id")
        .join(account_info, left_on="source_account_number", right_on="account_number")
        .join(account_holder, left_on="account_holder_id", right_on="id")
        .select(
            "transaction_id",
            "target_account_number",
            "created_on",
            "transaction_value",
            "is_cancelled",
            "source_account_number",
            "balance_taken_on",
            "balance",
            "full_name",
            "date_of_birth",
            "mobile_contact",
            "home_address_1",
        )
    )


if __name__ == "__main__":
    main()
