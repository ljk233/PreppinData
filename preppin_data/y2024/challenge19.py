"""2024: Week 19 - SuperBytes Sales and Profits

Inputs
------
- __input/2024/SuperBytes Sales_ Profits.xlsx

Outputs
-------
- output/2024/wk19_annual_sales_profits.ndjson
"""

import polars as pl


def solve(supabytes_sales_profits_fsrc: str) -> pl.DataFrame:
    """Solve challenge 19 of Preppin' Data 2024.

    Parameters
    ----------
    supabytes_sales_profits_fsrc : str
        Filepath of the Supabytes sales and profits data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the annual sales and profit data.
    """

    # Load and preprocess the data
    pre_data = preprocess_data(supabytes_sales_profits_fsrc)

    # Collect the output
    annual_sales_profit = pre_data.pipe(view_annual_sales_profit)

    return annual_sales_profit


def preprocess_data(supabytes_sales_profits_fsrc: str) -> pl.DataFrame:
    """Load and preprocess the Supabytes sales and profits data.

    Parameters
    ----------
    supabytes_sales_profits_fsrc : str
        Filepath of the Supabytes sales and profits data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed data.
    """

    data_dict = load_data(supabytes_sales_profits_fsrc)

    # Preprocess each DataFrame
    pre_data_ls = [
        data.pipe(annotate_data, sheet_name).pipe(clean_data)
        for sheet_name, data in data_dict.items()
    ]

    return pl.concat(pre_data_ls)


def load_data(supabytes_sales_profits_fsrc: str) -> dict[str, pl.DataFrame]:
    """Load data from an Excel file.

    Parameters
    ----------
    supabytes_sales_profits_fsrc : str

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary containing Polars DataFrames with loaded data.
        The keys are the names of the sheets in the Excel file,
        and the values are the corresponding DataFrames.
    """

    data_dict = pl.read_excel(supabytes_sales_profits_fsrc, sheet_id=0)

    return data_dict


def annotate_data(data: pl.DataFrame, sheet_name: str) -> pl.DataFrame:
    """Annotate the data with the source sheet name, which represents the
    year.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing data from a worksheet in the input Excel file.
    sheet_name : str
        Name of the worksheet. It is assumed that this represents the year.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the annotated data.
    """

    # Expressions
    year_expr = pl.lit(sheet_name).cast(pl.Int64)

    return data.with_columns(year=year_expr)


def clean_data(annotated_data: pl.DataFrame) -> pl.DataFrame:
    """Clean the annotated data.

    Parameters
    ----------
    annotated_data : pl.DataFrame
        DataFrame containing the annotated data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the cleansed data.
    """

    col_mapper = {"Sales": "sales", "Profits": "profits"}

    # Expressions
    quarter_expr = pl.col("").str.extract(r"(\d+)").cast(pl.Int64)

    value_num_expr = (
        pl.col("value").str.strip_chars_end("KMB").str.replace(",", "").cast(pl.Float64)
    )

    value_multiplier_expr = (
        pl.when(pl.col("value").str.ends_with("K"))
        .then(pl.lit(1_000))
        .when(pl.col("value").str.ends_with("M"))
        .then(pl.lit(1_000_000))
        .when(pl.col("value").str.ends_with("B"))
        .then(pl.lit(1_000_000_000))
        .otherwise(pl.lit(1))
    )

    value_expr = (value_num_expr * value_multiplier_expr).cast(pl.Int64)

    return (
        annotated_data.melt(id_vars=["year", ""])
        .with_columns(
            quarter=quarter_expr,
            value=value_expr,
        )
        .pivot(
            values="value",
            index=["year", "quarter"],
            columns="variable",
        )
        .rename(col_mapper)
    )


def view_annual_sales_profit(pre_data: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the quarterly sales and profit data by year.

    Parameters
    ----------
    pre_data : pl.DataFrame
        DataFrame containing the preprocessed data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the annual sales and profit data.
    """

    return pre_data.group_by("year").agg(
        sales=pl.sum("sales"), profits=pl.sum("profits")
    )
