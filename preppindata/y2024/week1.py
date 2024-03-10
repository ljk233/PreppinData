"""Prepping Data 2024 | Week 1

Preppin' Data 2024: Week 01 - Prep Air's Flow Card
https://preppindata.blogspot.com/2024/01/2024-week-1-prep-airs-flow-card.html

Task outline
============

1. Input the data
2. Split the Flight Details field to form:
  - Date 
  - Flight Number
  - From
  - To
  - Class
  - Price
3. Convert the following data fields to the correct data types:
  - Date to a date format
  - Price to a decimal value
4. Change the Flow Card field to Yes / No values instead of 1 / 0
5. Create two tables, one for Flow Card holders and one for non-Flow Card holders
6. Output the data sets

Modifications
=============

- All field names are set to snake_case
- A primary key is introduced to the data
- The "Flow Card?" field is set to a Boolean value, rather than Yes / No
- A single file is exported, rather than two individual files
"""

from typing import NamedTuple

import polars as pl


# Parameters
# ==========

RAW_PASSENGER_FLIGHT_DETAILS_CSV = "data/input/PD 2024 Wk 1 Input.csv"
PASSENGER_FLIGHT_DETAILS_NDJSON = "data/output/passenger_flight_details.ndjson"


# Classes
# =======


class PassengerFlightDetails(NamedTuple):
    flight_details: str = "flight_details"
    date: str = "date"
    flight_number: str = "flight_number"
    from_: str = "from"
    to: str = "to"
    class_: str = "class"
    price: str = "price"
    has_flow_card: str = "has_flow_card"
    number_of_bags_checked: str = "number_of_bags_checked"
    meal_type: str = "meal_type"


PFD = PassengerFlightDetails()


# Functions
# =========


def run_pipeline() -> None:
    """Run the pipeline.

    Reads the raw data, preprocesses and transforms it, and then writes
    the result to a .ndjson file.
    """

    try:
        # Load the raw data
        raw_data = pl.scan_csv(RAW_PASSENGER_FLIGHT_DETAILS_CSV)

        # Preprocess the raw data
        preprocessed_data = raw_data.pipe(preprocess_raw_data)

        # Transform the raw data
        transformed_data = preprocessed_data.pipe(transform_preprocessed_data)

        # Output transfromed
        transformed_data.write_ndjson(PASSENGER_FLIGHT_DETAILS_NDJSON)

        print("Pipeline executed successfully.")
    except Exception as e:
        print(f"Error during pipeline execution: {str(e)}")


def preprocess_raw_data(data: pl.LazyFrame) -> pl.LazyFrame:
    renamer = {
        "Flight Details": PFD.flight_details,
        "Flow Card?": PFD.has_flow_card,
        "Bags Checked": PFD.number_of_bags_checked,
        "Meal Type": PFD.meal_type,
    }

    return (
        data.with_row_index("id")
        .rename(renamer)
        .with_columns(pl.col(PFD.has_flow_card).cast(pl.Boolean))
    )


def transform_preprocessed_data(preprocessed_data: pl.LazyFrame) -> pl.DataFrame:
    return (
        preprocessed_data.pipe(extract_flight_details)
        .pipe(rename_flight_details_struct)
        .pipe(unnest_flight_details_struct)
        .collect()
    )


def extract_flight_details(data: pl.LazyFrame) -> pl.LazyFrame:
    flight_details_patt = r"(.+)//(.+)//(.+)-(.+)//(.+)//(.+)"

    return data.with_columns(
        pl.col(PFD.flight_details).str.extract_groups(flight_details_patt)
    )


def rename_flight_details_struct(data: pl.LazyFrame) -> pl.LazyFrame:
    flight_detail_struct_renamer = [
        PFD.date,
        PFD.flight_number,
        PFD.from_,
        PFD.to,
        PFD.class_,
        PFD.price,
    ]

    return data.with_columns(
        pl.col(PFD.flight_details).struct.rename_fields(flight_detail_struct_renamer)
    )


def unnest_flight_details_struct(data: pl.LazyFrame) -> pl.LazyFrame:
    return data.unnest(PFD.flight_details)
