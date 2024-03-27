"""2023: Week 29 - Moving Averages

See solution output at "output/2023/wk29_rolling_monthly_profit.ndjson".
"""

from datetime import date

import polars as pl


FORECAST_START_DATE = date(2023, 7, 1)


def solve(input_fsrc: str) -> pl.DataFrame:
    """Solve challenge 29 of Preppin' Data 2023."""
    # Preprocess the data
    pre_data = preprocess_data(input_fsrc)

    # Normalise the data
    store = pre_data.select("store").unique()
    bike_type = pre_data.select("bike_type").unique()
    date_map = pre_data.pipe(create_date_map)
    sales_in = date_map.pipe(create_sales_in, FORECAST_START_DATE)

    # Aggregate the data
    aggegrated_sales_data = (
        pre_data.join(date_map, on="sales_on")
        .join(sales_in, on="sales_in")
        .group_by("sales_in", "store", "bike_type")
        .agg(
            pl.sum("daily_sales").alias("monthly_sales"),
            pl.sum("daily_profit").alias("monthly_profit"),
        )
    )

    # Cross join the normalised data
    expected_sale_data = (
        store.join(bike_type, how="cross")
        .join(sales_in, how="cross")
        .filter(pl.col("is_forecast").not_())
        .select("store", "bike_type", "sales_in", "first_day_of_month")
    )

    # Calculate the rolling three month mean profit
    return (
        expected_sale_data.join(
            aggegrated_sales_data, on=["store", "bike_type", "sales_in"], how="left"
        )
        .fill_null(0)
        .sort("first_day_of_month")
        .with_columns(
            pl.col("monthly_profit")
            .rolling_mean(
                3,
                min_periods=3,
                by="first_day_of_month",
                warn_if_unsorted=False,
            )
            .over("store", "bike_type")
            .alias("rolling_monthly_profit")
        )
        .collect()
    )


def preprocess_data(input_fsrc: str) -> pl.LazyFrame:
    """Preprocess the source data."""
    renamer_dict = {
        "Date": "sales_on",
        "Store": "store",
        "Bike Type": "bike_type",
        "Sales": "daily_sales",
        "Profit": "daily_profit",
    }

    return pl.scan_csv(input_fsrc, try_parse_dates=True).rename(renamer_dict)


def create_date_map(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Create a LazyFrame that maps a date to its month and year."""
    return pre_data.select(
        pl.col("sales_on"), pl.col("sales_on").dt.strftime("%B-%Y").alias("sales_in")
    ).unique()


def create_sales_in(date_map: pl.LazyFrame, forecast_start_date: date) -> pl.LazyFrame:
    """Create a LazyFrame that contains each Month-Year, its first day
    of the month, and whether it represents an observation or a forecast.
    """
    first_day_of_month_expr = pl.min("sales_on").over("sales_in")
    is_forecast_date_pred_expr = pl.col("sales_on") >= forecast_start_date

    return date_map.select(
        "sales_in",
        first_day_of_month_expr.alias("first_day_of_month"),
        is_forecast_date_pred_expr.alias("is_forecast"),
    ).unique()
