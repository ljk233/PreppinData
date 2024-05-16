"""2024: Week 20 - SuperBytes Customer Data

Inputs
------
- __input/2024/SuperBytes Customer Data.xlsx

Outputs
-------
- output/2024/wk20_customer_day_of_week_analysis.json
"""

from collections import defaultdict
import json
from typing import Any

import polars as pl


def solve(supabytes_customer_fsrc: str) -> str:
    """Solves challenge 19 of Preppin' Data 2024.

    Parameters
    ----------
    supabytes_customer_fsrc : str
        Filepath of the input Excel workbook.

    Returns
    -------
    str
        A JSON-like string containing the results of the analysis.

    Notes
    -----
    This function takes the filepath of the input Excel workbook containing
    Supabytes customer data, preprocesses the data, performs analysis to
    rank customer spending by day of the week, and returns the results
    as a JSON-like string.
    """

    # Load the data
    pre_data_dict = preprocess_data(supabytes_customer_fsrc)

    # Unpack the data
    customer_spending = pre_data_dict[1]

    customer = pre_data_dict[2]

    # Collect the output
    customer_day_of_week_analysis = view_customer_day_of_week_analysis(
        customer_spending, customer
    )

    post_data_dicts = customer_day_of_week_analysis.pipe(postprocess_output, "weekday")

    return json.dumps(post_data_dicts, indent=2)


def preprocess_data(supabytes_customer_fsrc: str) -> dict[str, pl.DataFrame]:
    """Load and preprocess the Supabytes customer sales data.

    Parameters
    ----------
    supabytes_customer_fsrc : str
        Filepath of the input XLSX workbook.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary containing Polars DataFrames with preprocessed data.
        The keys are the names of the sheets in the Excel file, and the
        values are the corresponding DataFrames.
    """

    pre_data_dict = {}

    for sheet_id in range(1, 3):
        preprocess_func = get_preprocesser(sheet_id)

        pre_data_dict[sheet_id] = preprocess_func(supabytes_customer_fsrc)

    return pre_data_dict


def get_preprocesser(sheet_id: int):
    """Return the preprocessing function for the given sheet ID."""

    match sheet_id:
        case 1:
            return preprocess_customer_spending
        case 2:
            return preprocess_customer_names
        case _:
            raise ValueError(f"No matching processor for sheet_id={sheet_id}")


def preprocess_customer_spending(supabytes_customer_fsrc: str) -> pl.DataFrame:
    """Load and preprocess the Supabytes customer data.

    Parameters
    ----------
    supabytes_customer_fsrc : str
        Filepath of the input XLSX workbook.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocess customer spending data.
    """

    # Load the data
    data = load_customer_spending(supabytes_customer_fsrc)

    # Clean the data
    receipt_num_expr = pl.col("receipt_number").cast(pl.Int64)

    is_online_expr = (
        pl.when(pl.col("online") == "Yes").then(pl.lit(True)).otherwise(pl.lit(False))
    )

    return data.with_columns(receipt_num_expr, is_online=is_online_expr).drop(
        "online", "in_person"
    )


def load_customer_spending(supabytes_customer_fsrc: str) -> pl.DataFrame:
    """Load the customer spending data from the input Excel file.

    Parameters
    ----------
    supabytes_customer_fsrc : str
        Filepath of the input XLSX workbook.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the input customer spending data.
    """

    return pl.read_excel(
        supabytes_customer_fsrc,
        sheet_id=1,
        read_options={
            "has_header": False,
            "skip_rows": 2,
            "columns": range(8),
            "try_parse_dates": True,
            "new_columns": [
                "first_name",
                "last_name",
                "gender",
                "receipt_number",
                "purchased_on",
                "online",
                "in_person",
                "sale_value",
            ],
        },
    )


def preprocess_customer_names(supabytes_customer_fsrc: str) -> pl.DataFrame:
    """Load and preprocess the Supabytes customer names data.

    Parameters
    ----------
    supabytes_customer_fsrc : str
        Filepath of the input XLSX workbook.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocess customer names data.
    """

    # Load the data
    data = load_customer_names(supabytes_customer_fsrc)

    return data


def load_customer_names(supabytes_customer_fsrc: str) -> pl.DataFrame:
    """Load the customer names data from the input Excel file.

    Parameters
    ----------
    supabytes_customer_fsrc : str
        Filepath of the input XLSX workbook.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the input customer names data.
    """

    return pl.read_excel(
        supabytes_customer_fsrc,
        sheet_id=2,
        read_options={
            "new_columns": [
                "first_name",
                "last_name",
                "customer_id",
            ]
        },
    )


def view_customer_day_of_week_analysis(
    customer_spending: pl.DataFrame, customer: pl.DataFrame
) -> pl.DataFrame:
    """View the ranked customer spending by day of the week.

    Parameters
    ----------
    customer_spending : pl.DataFrame
        DataFrame containing the preprocessed customer spending data.
    customer : pl.DataFrame
        DataFrame containing the preprocessed customer names data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the customer spending analysis.
    """

    # Expressions
    weekday_expr = pl.col("purchased_on").dt.strftime("%A")

    rank_sale_value_expr = (
        pl.col("sale_value").rank("min", descending=True).over(weekday_expr)
    )

    weekday_num_expr = pl.col("purchased_on").dt.weekday()

    return (
        customer_spending.join(customer, on=["first_name", "last_name"])
        .with_columns(
            weekday=weekday_expr,
            rank=rank_sale_value_expr,
        )
        .sort(weekday_num_expr, "rank")
        .drop("first_name", "last_name", "purchased_on")
    )


def postprocess_output(
    data: pl.DataFrame, weekday_col: str
) -> dict[str, dict[str, Any]]:
    """Postprocesses the output data into a dictionary organized by weekdays.

    This function takes a Polars DataFrame containing the output data and
    organizes it into a dictionary where each key represents a weekday and
    its corresponding value is a list of dictionaries containing the data
    for that weekday.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the output data.
    weekday_col : str
        Column name containing the day of the week,

    Returns
    -------
    dict[str, dict[str, Any]]
        A dictionary where each key represents a weekday ('Monday', 'Tuesday',
        etc.) and its corresponding value is a list of dictionaries containing
        the data for that weekday.
    """

    data_dicts = data.to_dicts()

    post_data_dicts = defaultdict(lambda: [])

    for data_dict in data_dicts:
        weekday = data_dict[weekday_col]

        post_data_dicts[weekday].append(data_dict)

    return post_data_dicts
