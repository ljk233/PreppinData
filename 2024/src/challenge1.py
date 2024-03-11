"""week1.py

Pipeline for solving Challenge 1 of Preppin' Data 2024.
"""

from enum import auto, StrEnum

import polars as pl


class InputFields(StrEnum):
    FLIGHT_DETAILS = "Flight Details"
    FLOW_CARD = "Flow Card?"
    BAGS_CHECKED = "Bags Checked"
    MEAL_TYPE = "Meal Type"

    def __repr__(self) -> str:
        return self.value


class OutputFields(StrEnum):
    ID = auto()
    DATE = auto()
    FLIGHT_NUMBER = auto()
    FROM = auto()
    TO = auto()
    CLASS = auto()
    PRICE = auto()
    HAS_FLOW_CARD = auto()
    NUMBER_OF_BAGS_CHECKED = auto()
    MEAL_TYPE = auto()

    def __repr__(self) -> str:
        return self.value


def run_pipeline(input_data: pl.LazyFrame) -> pl.LazyFrame:
    """Run the pipeline for soliving Challenge 1."""
    return (
        input_data.pipe(add_primary_key)
        .pipe(set_input_data_types)
        .pipe(rename_input_fields)
        .pipe(extract_flight_details_groups)
        .pipe(rename_flight_details_fields)
        .pipe(unnest_flight_details_struct)
        .pipe(set_flight_details_data_types)
    )


def run_pipeline_with_fix(input_data: pl.LazyFrame) -> pl.LazyFrame:
    """Run the pipeline with the fix to the class field assignments."""
    return input_data.pipe(run_pipeline).pipe(fix_class_field)


def add_primary_key(data: pl.LazyFrame) -> pl.LazyFrame:
    """Add a primary key to the input data.

    Precursors: None
    """
    return data.with_row_index(OutputFields.ID)


def set_input_data_types(data: pl.LazyFrame) -> pl.LazyFrame:
    """Set the data types of the input data."""
    cast_flow_card_expr = pl.col(InputFields.FLOW_CARD).cast(pl.Boolean)

    return data.with_columns(cast_flow_card_expr)


def rename_input_fields(data: pl.LazyFrame) -> pl.LazyFrame:
    """Rename the input data fields to snake_case."""
    renamer = {
        InputFields.FLOW_CARD: OutputFields.HAS_FLOW_CARD,
        InputFields.BAGS_CHECKED: OutputFields.NUMBER_OF_BAGS_CHECKED,
        InputFields.MEAL_TYPE: OutputFields.MEAL_TYPE,
    }

    return data.rename(renamer)


def extract_flight_details_groups(data: pl.LazyFrame) -> pl.LazyFrame:
    """Extract the groups from the flight details column into a struct."""
    flight_details_patt = r"(.+)//(.+)//(.+)-(.+)//(.+)//(.+)"
    extract_groups_expr = pl.col(InputFields.FLIGHT_DETAILS).str.extract_groups(
        flight_details_patt
    )

    return data.with_columns(extract_groups_expr)


def rename_flight_details_fields(data: pl.LazyFrame) -> pl.LazyFrame:
    """Rename the columns in the flight details struct."""
    renamer = [
        OutputFields.DATE,
        OutputFields.FLIGHT_NUMBER,
        OutputFields.FROM,
        OutputFields.TO,
        OutputFields.CLASS,
        OutputFields.PRICE,
    ]

    flight_details_struct_rename_expr = pl.col(
        InputFields.FLIGHT_DETAILS
    ).struct.rename_fields(renamer)

    return data.with_columns(flight_details_struct_rename_expr)


def unnest_flight_details_struct(data: pl.LazyFrame) -> pl.LazyFrame:
    """Unnest the flight details struct so each field has its own column."""
    return data.unnest(InputFields.FLIGHT_DETAILS)


def set_flight_details_data_types(data: pl.LazyFrame) -> pl.LazyFrame:
    """Set the data types of the final output data."""
    cast_date_expr = pl.col(OutputFields.DATE).str.to_date()
    cast_price_expr = pl.col(OutputFields.PRICE).cast(pl.Float64)

    return data.with_columns(cast_date_expr, cast_price_expr)


def fix_class_field(data: pl.LazyFrame) -> pl.LazyFrame:
    """Fix the assignments for the "class" field."""
    replacer_dict = {
        "Economy": "First Class",
        "First Class": "Economy",
        "Business Class": "Premium Economy",
        "Premium Economy": "Business Class",
    }

    class_replacer_expr = pl.col(OutputFields.CLASS).str.replace_many(
        list(replacer_dict.keys()), list(replacer_dict.values())
    )

    return data.with_columns(class_replacer_expr)
