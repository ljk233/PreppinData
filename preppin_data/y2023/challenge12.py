"""2023: Week 12 - Regulatory Reporting Alignment

See solution output at "output/2023/wk12_new_customers_ready_to_report.ndjson".
"""

from datetime import date

import polars as pl


def solve(input_data_str: str) -> pl.LazyFrame:
    """Solve challenge 12 of Preppin' Data 2023."""
    # Preprocess the source data
    uk_bank_holiday = preprocess_uk_bank_holiday(input_data_str)
    uk_new_customers = preprocess_uk_new_customer(input_data_str)
    roi_new_customer = preprocess_roi_new_customer(input_data_str)

    # Construct the UK reporting calendar
    st_date = date(2021, 12, 31)
    end_date = date(2023, 12, 31)
    uk_reporting_calendar = constuct_reporting_calendar(
        st_date, end_date, uk_bank_holiday
    )

    # Construct the reporting detail table
    uk_reporting_calendar_detail = (
        uk_reporting_calendar.select("reported_on")
        .unique()
        .pipe(construct_report_calendar_detail)
    )

    # Aggregate the number of new customers per reporting day
    new_customers_per_reporting_day = (
        uk_new_customers.join(
            roi_new_customer,
            left_on="joined_on",
            right_on="roi_reported_on",
            how="left",
        )
        .join(uk_reporting_calendar, on="joined_on")
        .group_by("reported_on")
        .agg(
            pl.max("roi_reporting_month"),
            pl.sum("number_of_new_uk_customers"),
            pl.sum("number_of_new_roi_customers"),
        )
    )

    # Complete the transformation by enriching the reporting detail
    return (
        new_customers_per_reporting_day.join(
            uk_reporting_calendar_detail, on="reported_on", how="left"
        )
        .drop_nulls("reporting_month")
        .with_columns(
            pl.when(pl.col("reporting_month") == pl.col("roi_reporting_month"))
            .then(pl.lit(False))
            .otherwise(pl.lit(True))
            .alias("misalignment_flag")
        )
        .select(
            "misalignment_flag",
            pl.col("reporting_month").alias("uk_reporting_month"),
            "reporting_day",
            "reported_on",
            "number_of_new_uk_customers",
            "number_of_new_roi_customers",
            "roi_reporting_month",
        )
    )


def preprocess_uk_bank_holiday(fsrc: str) -> pl.LazyFrame:
    """Preprocess the UK Bank holiday source data."""
    return (
        pl.read_excel(fsrc, sheet_name="UK Bank Holidays")
        .lazy()
        .with_columns(pl.col("Year").forward_fill().cast(pl.Utf8))
        .drop_nulls()
        .with_columns(pl.col("Date") + "-" + pl.col("Year"))
        .select(
            pl.col("Date").str.to_date("%d-%B-%Y").alias("bank_holiday_date"),
            pl.col("Bank holiday").alias("bank_holiday_name"),
        )
    )


def preprocess_uk_new_customer(fsrc: str) -> pl.LazyFrame:
    """Preprocess the (UK) New Customer source data."""
    return (
        pl.read_excel(fsrc, sheet_name="New Customers")
        .lazy()
        .select(
            pl.col("Date").str.to_date("%m-%d-%y").alias("joined_on"),
            pl.col("New Customers").alias("number_of_new_uk_customers"),
        )
    )


def preprocess_roi_new_customer(fsrc: str) -> pl.LazyFrame:
    """Preprocess the ROI New Customer source data."""
    return (
        pl.read_excel(fsrc, sheet_name="ROI New Customers")
        .lazy()
        .select(
            pl.col("Reporting Date").str.to_date("%m-%d-%y").alias("roi_reported_on"),
            pl.col("Reporting Month").alias("roi_reporting_month"),
            pl.col("New Customers").alias("number_of_new_roi_customers"),
        )
    )


def constuct_reporting_calendar(
    st_date: date, end_date: date, bank_holiday: pl.LazyFrame
) -> pl.LazyFrame:
    """Construct the reporting calendar, which maps the date a customer
    joined to their first available reporting date.
    """
    # Gather all dates in relavent dates
    calendar = pl.LazyFrame().with_columns(
        pl.date_range(st_date, end_date).alias("date")
    )

    # Gather candidate reporting dates
    candidate_reporting_date = (
        calendar.rename({"date": "reported_on"})
        # Remove bank holidays
        .join(
            bank_holiday,
            left_on="reported_on",
            right_on="bank_holiday_date",
            how="anti",
        )
        # Filter weekends
        .filter(pl.col("reported_on").dt.weekday() <= 5)
    )

    return (
        calendar.rename({"date": "joined_on"})
        .join(
            candidate_reporting_date,
            left_on="joined_on",
            right_on="reported_on",
            how="cross",
        )
        # Remove invalid reporting dates (they cannot be in the past)
        .filter(pl.col("joined_on") <= pl.col("reported_on"))
        # Select the minimum reporting date after a customer joins
        .group_by("joined_on")
        .agg(pl.min("reported_on"))
    )


def construct_report_calendar_detail(reporting_calendar: pl.LazyFrame) -> pl.LazyFrame:
    """Construct the reporting calendar detail data, which maps a reporting
    date to its reporting month and day.
    """
    return (
        reporting_calendar.join(reporting_calendar, on="reported_on", how="cross")
        .filter(pl.col("reported_on_right") > pl.col("reported_on"))
        .group_by("reported_on")
        .agg(pl.min("reported_on_right"))
        .sort("reported_on")
        .with_columns(pl.col("reported_on_right").dt.strftime("%b-%y"))
        .select(
            "reported_on",
            pl.col("reported_on_right").alias("reporting_month"),
            pl.cum_count("reported_on_right")
            .over("reported_on_right")
            .alias("reporting_day"),
        )
    )
