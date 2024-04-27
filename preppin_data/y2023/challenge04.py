"""2023: Week 4 - New Customers

Inputs
------
- __input/2023/New Customers.xlsx

Outputs
-------
- output/2023/wk04_new_customer.ndjson
"""

import polars as pl


def solve(new_customers_fsrc: str) -> pl.DataFrame:
    """Solve challenge 4 of Preppin' Data 2023.

    Parameters
    ----------
    new_customers_fsrc : str
        Filepath of the input XLSX file containing new customer data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed new customer data.

    Notes
    -----
    This function performs the following steps:
    1. Load the new customer data from the provided XLSX file.
    2. Preprocess the new customer data by harmonizing column names, reshaping
       the data, and cleaning it.
    """

    # Load the new customer data
    data_dict = load_new_customer_data(new_customers_fsrc)

    # Preprocess the new customer data
    pre_new_customer = preprocess_new_customer_data(data_dict)

    return pre_new_customer


def load_new_customer_data(fsrc: str) -> dict[str, pl.DataFrame]:
    """Load new customer data from an Excel file.

    Parameters
    ----------
    fsrc : str
        Filepath of the input Excel file containing new customer data.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary containing sheet names as keys and corresponding DataFrames
        as values.
    """
    data = pl.read_excel(fsrc, sheet_id=0)

    return data


def preprocess_new_customer_data(data_dict: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Preprocess the new customer data.

    Parameters
    ----------
    data_dict : dict[str, pl.DataFrame]
        A dictionary containing sheet names as keys and corresponding DataFrames
        as values.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed new customer data.

    Notes
    -----
    This function performs the following actions:
    - Harmonizes column names.
    - Reshapes the data.
    - Cleans the data, including renaming columns and formatting dates.

    Primary key is {person_id}
    """

    data_arr = []

    for sheet_name, data in data_dict.items():
        data_arr.append(
            data.pipe(harmonize_new_customer_data)
            .pipe(reshape_new_customer_data)
            .pipe(clean_new_customer_data, sheet_name)
        )

    return pl.concat(data_arr)


def harmonize_new_customer_data(data: pl.DataFrame) -> pl.DataFrame:
    """Harmonize column names of the new customer data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the new customer data.

    Returns
    -------
    pl.DataFrame
        DataFrame with harmonized column names.

    Notes
    -----
    This function standardizes column names to ensure consistency across
    the dataset.
    """

    col_mapper = {
        orig_col: "Demographic" for orig_col in data.columns if orig_col.startswith("D")
    }

    return data.rename(col_mapper)


def reshape_new_customer_data(harmonized_data: pl.DataFrame) -> pl.DataFrame:
    """Reshape the new customer data.

    Parameters
    ----------
    harmonized_data : pl.DataFrame
        DataFrame containing the harmonized new customer data.

    Returns
    -------
    pl.DataFrame
        DataFrame with reshaped new customer data.
    """

    return harmonized_data.pivot(
        values="Value",
        columns="Demographic",
        index=["ID", "Joining Day"],
    )


def clean_new_customer_data(
    reshaped_data: pl.DataFrame, sheet_name: str
) -> pl.DataFrame:
    """Clean the reshaped new customer data.

    Parameters
    ----------
    reshaped_data : pl.LazyFrame
        DataFrame containing the reshaped new customer data.
    sheet_name : str
        Name of the sheet from which the data was extracted.

    Returns
    -------
    pl.DataFrame
        DataFrame with cleaned new customer data.

    Notes
    -----
    This function performs data cleaning tasks such as renaming columns
    and formatting dates.
    """

    col_mapper = {
        "ID": "person_id",
        "Ethnicity": "ethnicity",
        "Date of Birth": "born_on",
        "Account Type": "account_type",
    }

    # Expressions
    month_year_lit = pl.lit(f"-{sheet_name}-2023")

    joined_on_expr = (pl.col("Joining Day").cast(pl.Utf8) + month_year_lit).str.to_date(
        "%d-%B-%Y"
    )

    date_of_birth_expr = pl.col("Date of Birth").str.to_date("%m/%d/%Y")

    return (
        reshaped_data.with_columns(date_of_birth_expr, joined_on=joined_on_expr)
        .sort("joined_on")
        .unique("ID")
        .drop("Joining Day")
        .rename(col_mapper)
    )
