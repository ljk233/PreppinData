"""week1.py

Pipeline for solving Challenge 2 of Preppin' Data 2024.
"""

from enum import auto, StrEnum

import polars as pl

from . import challenge1

INPUT_FIELDS = challenge1.OutputFields


class OutputFields(StrEnum):
    HAS_FLOW_CARD = auto()
    QUARTER = auto()
    ECONOMY_CLASS = auto()
    PREMIUM_ECONOMY_CLASS = auto()
    BUSINESS_CLASS = auto()
    FIRST_CLASS = auto()
    PRICE_AGGREGATE = auto()

    def __repr__(self) -> str:
        return self.value


def run_pipeline(input_data: pl.LazyFrame) -> pl.LazyFrame:
    """Run the pipeline for soliving Challenge 2."""
    return (
        input_data.pipe(challenge1.run_pipeline_with_fix)
        .pipe(preprocess_data)
        .pipe(append_date_quarter)
        .pipe(aggregate_price_data)
        .collect()
        .pipe(rotate_aggregated_price_data)
        .lazy()
        .pipe(postprocess_data)
    )


def preprocess_data(input_data: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocess the input data by keeping fields relavent to the task."""
    return input_data.with_columns(
        INPUT_FIELDS.DATE,
        INPUT_FIELDS.HAS_FLOW_CARD,
        INPUT_FIELDS.CLASS,
        INPUT_FIELDS.PRICE,
    )


def append_date_quarter(data: pl.LazyFrame) -> pl.LazyFrame:
    """Append the date quarter to the data."""
    date_to_qtr_expr = (
        pl.col(INPUT_FIELDS.DATE).dt.quarter().alias(OutputFields.QUARTER)
    )

    return data.with_columns(date_to_qtr_expr)


def aggregate_price_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate the quarterly price data by flow card and seat class."""
    groups = [OutputFields.HAS_FLOW_CARD, OutputFields.QUARTER, INPUT_FIELDS.CLASS]
    price_col = pl.col(INPUT_FIELDS.PRICE)
    min_price = price_col.min()
    median_price = price_col.median()
    max_price = price_col.max()

    return data.group_by(groups).agg(
        min_price.alias("min_price"),
        median_price.alias("median_price"),
        max_price.alias("max_price"),
    )


def rotate_aggregated_price_data(data: pl.DataFrame) -> pl.DataFrame:
    """Rotate the aggregated price data."""
    melt_id_vars = [
        OutputFields.HAS_FLOW_CARD,
        OutputFields.QUARTER,
        INPUT_FIELDS.CLASS,
    ]
    pivot_index = [
        OutputFields.HAS_FLOW_CARD,
        OutputFields.QUARTER,
        OutputFields.PRICE_AGGREGATE,
    ]

    return data.melt(
        id_vars=melt_id_vars, variable_name=OutputFields.PRICE_AGGREGATE
    ).pivot(index=pivot_index, columns=INPUT_FIELDS.CLASS, values="value")


def postprocess_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Postprocess the data."""
    renamer_dict = {
        "Economy": OutputFields.ECONOMY_CLASS,
        "Premium Economy": OutputFields.PREMIUM_ECONOMY_CLASS,
        "Business Class": OutputFields.BUSINESS_CLASS,
        "First Class": OutputFields.FIRST_CLASS,
    }

    return data.rename(renamer_dict).select(
        OutputFields.HAS_FLOW_CARD,
        OutputFields.QUARTER,
        OutputFields.PRICE_AGGREGATE,
        OutputFields.ECONOMY_CLASS,
        OutputFields.PREMIUM_ECONOMY_CLASS,
        OutputFields.BUSINESS_CLASS,
        OutputFields.FIRST_CLASS,
    )
