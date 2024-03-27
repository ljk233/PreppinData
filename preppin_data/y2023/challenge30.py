"""2023: Week 30 - % Difference From

See solution output at "output/2023/wk30_financial_quarterly_analysis.ndjson".
"""

import polars as pl


COLUMN_MAP = pl.DataFrame(
    [
        ("id", "id"),
        ("*BikeType", "bike_type"),
        ("*Bike Type", "bike_type"),
        ("*Model", "bike_model"),
        ("*Order Date", "ordered_on"),
        ("*Sales", "volume_sold"),
        ("*Store", "store_location"),
        ("Bike Type", "bike_type"),
        ("BikeType", "bike_type"),
        ("Model", "bike_model"),
        ("Order Date", "ordered_on"),
        ("OrderDate", "ordered_on"),
        ("Sales", "volume_sold"),
        ("Store", "store_location"),
    ],
    schema=["orig_column_name", "new_column_name"],
)

FISCAL_QUARTER_MAP = pl.DataFrame(
    [(1, 3), (2, 4), (3, 1), (4, 2)],
    schema={"calendar_quarter": pl.Int8, "fiscal_quarter": pl.Int8},
)


def solve(input_fsrc: str) -> pl.DataFrame:
    """Solve challenge 30 of Preppin' Data 2023."""
    # Load the data
    data_dict = pl.read_excel(input_fsrc, sheet_id=0)

    # Preprocess the data
    pre_data_ls = [
        preprocess_data(data, month_name, COLUMN_MAP)
        for month_name, data in data_dict.items()
    ]

    # Transform the data
    return (
        pl.concat(pre_data_ls)
        .pipe(aggregate_data, FISCAL_QUARTER_MAP)
        .pipe(running_difference)
    )


def preprocess_data(
    data: pl.DataFrame, month_name: str, column_map: pl.DataFrame
) -> pl.DataFrame:
    """Preprocess the data."""
    return data.pipe(harmonize_data, column_map).pipe(clean_data, month_name)


def harmonize_data(data: pl.DataFrame, column_map: pl.DataFrame) -> pl.DataFrame:
    """Harmonize the data by mapping each column name to it's proper snake_case
    equivalent.
    """
    return (
        data.with_row_index("id", offset=1)
        .melt(id_vars="id", variable_name="orig_column_name")
        .join(column_map, on=["orig_column_name"])
        .pivot(values="value", index="id", columns="new_column_name")
        .drop("id")
    )


def clean_data(harmonized_data: pl.DataFrame, month_name: str) -> pl.DataFrame:
    """Clean the harmonized data."""
    #  Expression
    extract_day_year_expr = (
        pl.col("ordered_on")
        .str.extract_groups(r"(\d+), (\d+)$")
        .struct.rename_fields(["day", "year"])
    )

    concat_ordered_on_expr = (
        pl.col("ordered_on").struct["day"]
        + month_name
        + pl.col("ordered_on").struct["year"]
    )

    parse_ordered_on_expr = pl.col("ordered_on").str.to_date("%d%B%Y")

    return (
        harmonized_data.with_columns(
            extract_day_year_expr, pl.col("volume_sold").cast(pl.Int64)
        )
        .with_columns(concat_ordered_on_expr.alias("ordered_on"))
        .with_columns(parse_ordered_on_expr)
    )


def aggregate_data(
    pre_data: pl.DataFrame, fisqual_quarter: pl.DataFrame
) -> pl.DataFrame:
    """Aggregate the sales per store to the fiscal quarter level."""
    # Expressions
    quarter_expr = pl.col("ordered_on").dt.quarter()

    return (
        pre_data.with_columns(quarter_expr.alias("calendar_quarter"))
        .join(fisqual_quarter, on="calendar_quarter")
        .group_by("store_location", "fiscal_quarter")
        .agg(pl.sum("volume_sold"))
    )


def running_difference(agg_data: pl.DataFrame) -> pl.DataFrame:
    """Calculate the running percentage difference of volume sales by store
    location.
    """
    pct_diff_expr = 100 * pl.col("volume_sold").pct_change().over("store_location")

    return agg_data.sort("store_location", "fiscal_quarter").with_columns(
        pct_diff_expr.round(1).alias("pct_diff")
    )
