"""2024: Week 3 - Average Price Analysis

Inputs
======
- __input/2024/PD 2024 Wk 1 Input.csv
- __input/2024/PD 2024 Wk 3 Input.xlsx

Outputs
=======
- output/2024/wk03_sales_against_target.ndjson
"""

import polars as pl

from .challenge01 import preprocess_fixed_flight_detail_data


def solve(pd_input_wk1_fsrc: str, pd_input_wk3_fsrc: str) -> pl.DataFrame:
    """Solve challenge 3 of Preppin' Data 2024.

    Parameters
    ----------
    pd_input_wk1_fsrc : str
        Filepath of the input CSV file for Week 1.
    pd_input_wk3_fsrc : str
        Filepath of the input CSV file for Week 3.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the performance analysis against the target
        sales data.
    """

    # Preprocess the data
    pre_flight_detail = preprocess_fixed_flight_detail_data(pd_input_wk1_fsrc)

    pre_target_sales = preprocess_target_sales_data(pd_input_wk3_fsrc, 2024)

    # Collect the output
    performance_against_target = view_performance_against_target(
        pre_flight_detail, pre_target_sales
    )

    return performance_against_target


def preprocess_target_sales_data(
    pd_input_wk3_fsrc: str, calendar_year: int
) -> pl.DataFrame:
    """Preprocess the target sales data.

    Parameters
    ----------
    pd_input_wk3_fsrc : str
        Filepath of the input CSV file for Week 3.
    calendar_year : int
        Calendar year for the target sales data.

    Returns
    -------
    pl.DataFrame
        DataFrame representing the preprocessed target sales data.
    """

    data_dict = load_target_sales_data(pd_input_wk3_fsrc)

    data = merge_target_sales_data(data_dict)

    return data.pipe(clean_target_sales_data, calendar_year).pipe(
        harmonize_target_sales_data
    )


def load_target_sales_data(fsrc: str) -> dict[str, pl.DataFrame]:
    """Load data from the input XLSX file.

    Parameters
    ----------
    fsrc : str
        Filepath of the input XLSX file.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary where keys are worksheet names and values are DataFrames.
        Each DataFrame contains the data loaded from a worksheet in the Excel
        file.
    """

    return pl.read_excel(fsrc, sheet_id=0)


def merge_target_sales_data(data_dict: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Merge target sales data from multiple worksheets.

    Parameters
    ----------
    data_dict : dict[str, pl.DataFrame]
        A dictionary containing DataFrames from different worksheets.

    Returns
    -------
    pl.DataFrame
        DataFrame containing merged target sales data.
    """

    return pl.concat(data_dict.values())


def clean_target_sales_data(
    merged_data: pl.DataFrame, calendar_year: int
) -> pl.DataFrame:
    """Clean the target sales data.

    Parameters
    ----------
    reshaped_data : pl.DataFrame
        Target sales data as a Polars DataFrame.
    calendar_year : int
        Calendar year for the target sales data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the cleansed target sales data.
    """

    col_mapper = {"Class": "seat_class_code", "Target": "target_sales"}

    # Expressions
    target_month_expr = month_starts_on_expr("Month", calendar_year).dt.strftime(
        "%Y-%m"
    )

    return (
        merged_data.with_columns(target_month=target_month_expr)
        .drop("Month")
        .rename(col_mapper)
    )


def harmonize_target_sales_data(cleaned_data: pl.DataFrame) -> pl.DataFrame:
    """Harmonize the target sales data with the actual sales data.

    Parameters
    ----------
    cleaned_data : pl.DataFrame
        DataFrame containing the cleansed target sales data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the harmonized target sales data.
    """

    seat_class_map = {
        "E": "Economy",
        "PE": "Premium Economy",
        "BC": "Business Class",
        "FC": "First Class",
    }

    seat_class_expr = pl.col("seat_class_code").replace(seat_class_map)

    return cleaned_data.with_columns(seat_class=seat_class_expr)


def month_starts_on_expr(month_col: str, calendar_year: int) -> pl.Expr:
    """Return an Expression to represent the data the month starts on as
    a DataTime.
    """

    calendar_year_str = f"{calendar_year}-"

    month_expr = pl.col(month_col).cast(pl.Utf8)

    starts_on_expr = calendar_year_str + month_expr + "-01"

    return starts_on_expr.str.to_date()


def view_performance_against_target(
    pre_flight_detail: pl.DataFrame, pre_target_sales: pl.DataFrame
) -> pl.DataFrame:
    """View the performance against target sales.

    Parameters
    ----------
    pre_flight_detail : pl.DataFrame
        DataFrame containing pre-processed flight detail data.
    pre_target_sales : pl.DataFrame
        DataFrame containing pre-processed target sales data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing performance metrics against target sales.
    """

    # Aggregate the total monthly sales data
    total_monthly_sales = pre_flight_detail.pipe(view_total_montly_sales)

    # Expressions
    different_to_target_expr = pl.col("total_sales") - pl.col("target_sales")

    return pre_target_sales.join(  # .join(SEAT_CLASS, on="seat_class_code")
        total_monthly_sales,
        left_on=["seat_class", "target_month"],
        right_on=["seat_class", "flight_month"],
    ).select(
        "seat_class",
        "target_month",
        "total_sales",
        "target_sales",
        difference_to_target=different_to_target_expr,
    )


def view_total_montly_sales(pre_flight_detail: pl.LazyFrame) -> pl.DataFrame:
    """View the total monthly sales by seat classs.

    Parameters
    ----------
    pre_flight_detail : pl.DataFrame
        DataFrame containing flight detail price data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the total monthly sales by seat classs.
    """

    flight_month_expr = pl.col("flew_on").dt.strftime("%Y-%m")

    return (
        pre_flight_detail.group_by(
            "seat_class",
            flight_month=flight_month_expr,
        )
        .agg(total_sales=pl.sum("price"))
        .collect()
    )
