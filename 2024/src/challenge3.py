"""week1.py

Pipeline for solving Challenge 3 of Preppin' Data 2024.
"""

from enum import auto, StrEnum

import polars as pl

from . import challenge1

WEEK1_INPUT_FIELDS = challenge1.OutputFields


class Week3InputFields(StrEnum):
    MONTH = "Month"
    CLASS = "Class"
    TARGET = "Target"

    def __repr__(self) -> str:
        return self.value


class OutputFields(StrEnum):
    MONTH = auto()
    CLASS = auto()
    TOTAL_SALES = auto()
    TARGET_SALES = auto()
    DIFFERENCE_TO_TAGET = auto()

    def __repr__(self) -> str:
        return self.value


def run_pipeline(
    week1_input_data: pl.LazyFrame, week3_input_data_dict: dict[str, pl.DataFrame]
) -> pl.LazyFrame:
    """"""
    # Collect the flight details data
    flight_details_data = week1_input_data.pipe(challenge1.run_pipeline_with_fix)

    # Gather the monthly aggregated sales data
    monthly_total_sales_data = flight_details_data.pipe(run_aggregate_sales_pipeline)

    # Gather the monthly target sales data
    monthly_target_sales_data = run_monthly_target_sales_pipeline(week3_input_data_dict)

    return (
        join_sales_target(monthly_total_sales_data, monthly_target_sales_data)
        .pipe(append_difference_to_sales)
        .pipe(postprocess_output_data)
    )


def run_aggregate_sales_pipeline(flight_details_data: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate the flight details data, calculating the total sales per
    class and month.
    """
    return flight_details_data.pipe(append_sales_month_field).pipe(aggregate_sales_data)


def run_monthly_target_sales_pipeline(
    week3_input_data_dict: dict[str, pl.DataFrame]
) -> pl.LazyFrame:
    """Process the data dictionary containing the targets sales data, which
    is the input for week 3.
    """
    return (
        unpack_and_stack_week3_data(week3_input_data_dict)
        .pipe(preprocess_targetsdata)
        .pipe(harmonise_targets_class_field)
    )


def unpack_and_stack_week3_data(
    input_data_dict: dict[str, pl.DataFrame]
) -> pl.LazyFrame:
    return pl.concat(input_data_dict.values()).lazy()


def preprocess_targetsdata(input_data: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocess the input data dictionary after reading the input data
    from Week 3.
    """
    renamer_dict = {
        Week3InputFields.MONTH: OutputFields.MONTH,
        Week3InputFields.TARGET: OutputFields.TARGET_SALES,
        Week3InputFields.CLASS: OutputFields.CLASS,
    }

    return input_data.rename(renamer_dict)


def harmonise_targets_class_field(targets_data: pl.LazyFrame) -> pl.LazyFrame:
    """Harmonise the levels of the class field with that of the sales data."""
    harmonise_class_levels_expr = pl.col(OutputFields.CLASS).str.replace_many(
        ["E", "PE", "BC", "FC"],
        ["Economy", "Premium Economy", "Business Class", "First Class"],
    )

    return targets_data.with_columns(harmonise_class_levels_expr)


def append_sales_month_field(sales_data: pl.LazyFrame) -> pl.LazyFrame:
    date_to_month_expr = (
        pl.col(WEEK1_INPUT_FIELDS.DATE).dt.month().alias(OutputFields.MONTH)
    ).cast(pl.Int64)

    return sales_data.with_columns(date_to_month_expr)


def aggregate_sales_data(sales_data: pl.LazyFrame) -> pl.DataFrame:
    groups = [OutputFields.CLASS, OutputFields.MONTH]
    sum_price_expr = pl.sum(WEEK1_INPUT_FIELDS.PRICE).alias(OutputFields.TOTAL_SALES)

    return sales_data.group_by(groups).agg(sum_price_expr)


def join_sales_target(
    sales_data: pl.LazyFrame, targets_data: pl.LazyFrame
) -> pl.LazyFrame:
    return sales_data.join(targets_data, on=[OutputFields.MONTH, OutputFields.CLASS])


def append_difference_to_sales(sales_targets_data: pl.LazyFrame) -> pl.LazyFrame:
    diff_to_sales_expr = (
        pl.col(OutputFields.TOTAL_SALES) - pl.col(OutputFields.TARGET_SALES)
    ).alias(OutputFields.DIFFERENCE_TO_TAGET)

    return sales_targets_data.with_columns(diff_to_sales_expr)


def postprocess_output_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Postprocess the output data."""

    return data.select(
        OutputFields.DIFFERENCE_TO_TAGET,
        OutputFields.MONTH,
        OutputFields.TOTAL_SALES,
        OutputFields.CLASS,
        OutputFields.TARGET_SALES,
    )
