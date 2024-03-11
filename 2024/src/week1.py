"""week1.py

Functions for cleaning the input data for Week 1 of Preppin' Data 2024.
"""

import polars as pl

from .schema import week1


INPUT_SCHEMA = week1.InputSchema()
OUTPUT_SCHEMA = week1.OutputSchema()


def run_pipeline(data: pl.LazyFrame) -> pl.LazyFrame:
    """Run the preprocessing and transformation pipelines."""
    return (
        data.pipe(add_primary_key)
        .pipe(set_input_data_types)
        .pipe(rename_input_fields)
        .pipe(extract_flight_details_groups)
        .pipe(rename_flight_details_fields)
        .pipe(unnest_flight_details_struct)
    )


def add_primary_key(data: pl.LazyFrame) -> pl.LazyFrame:
    """Add a primary key to the input data.

    Precursors: None
    """
    return data.with_row_index(OUTPUT_SCHEMA.id_)


def set_input_data_types(data: pl.LazyFrame) -> pl.LazyFrame:
    """Set the data types.

    Precursors: None
    """
    cast_flow_card_expr = pl.col(INPUT_SCHEMA.flow_card).cast(pl.Boolean)

    return data.with_columns(cast_flow_card_expr)


def rename_input_fields(data: pl.LazyFrame) -> pl.LazyFrame:
    """Rename the input data fields to snake_case."""
    renamer = {
        INPUT_SCHEMA.flight_details: OUTPUT_SCHEMA.flight_details,
        INPUT_SCHEMA.flow_card: OUTPUT_SCHEMA.has_flow_card,
        INPUT_SCHEMA.bags_checked: OUTPUT_SCHEMA.number_of_bags_checked,
        INPUT_SCHEMA.meal_type: OUTPUT_SCHEMA.meal_type,
    }

    return data.rename(renamer)


def extract_flight_details_groups(data: pl.LazyFrame) -> pl.LazyFrame:
    """Extract the groups from the flight details column into a struct."""
    flight_details_patt = r"(.+)//(.+)//(.+)-(.+)//(.+)//(.+)"

    return data.with_columns(
        pl.col("flight_details").str.extract_groups(flight_details_patt)
    )


def rename_flight_details_fields(data: pl.LazyFrame) -> pl.LazyFrame:
    """Rename the columns in the flight details struct."""
    renamer = [
        OUTPUT_SCHEMA.date,
        OUTPUT_SCHEMA.flight_number,
        OUTPUT_SCHEMA.from_,
        OUTPUT_SCHEMA.to,
        OUTPUT_SCHEMA.class_,
        OUTPUT_SCHEMA.price,
    ]

    flight_details_struct_rename_expr = pl.col("flight_details").struct.rename_fields(
        renamer
    )

    return data.with_columns(flight_details_struct_rename_expr)


def unnest_flight_details_struct(data: pl.LazyFrame) -> pl.LazyFrame:
    """Unnest the flight details struct so each field has its own column."""
    return data.unnest(OUTPUT_SCHEMA.flight_details)
