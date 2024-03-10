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

import polars as pl


# Parameters
# ==========

PASSENGER_FLIGHT_DETAILS_CSV = "data/input/PD 2024 Wk 1 Input.csv"
PASSENGER_FLIGHT_DETAIL_NDJSON = "data/output/passenger_flight_details.ndjson"


# Functions
# =========


def run_pipeline() -> None:
    """Run the pipeline."""

    raw_data = pl.scan_csv(PASSENGER_FLIGHT_DETAILS_CSV)

    preproc_data = raw_data.pipe(preprocess_data)
    transformed_data = preproc_data.pipe(transform_data)

    transformed_data.collect().write_ndjson(PASSENGER_FLIGHT_DETAIL_NDJSON)


def preprocess_data(raw_data: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocess the data.

    Parameters
    ----------
    raw_data : pl.LazyFrame
        The raw data.

    Returns
    -------
    pl.LazyFrame
        Preprocessed data.

    Notes
    -----
    We introduce a primary key into the data, and clean up the field names.
    """
    return raw_data.with_row_index("id", offset=1).rename(
        {
            "Flight Details": "flight_details",
            "Flow Card?": "has_flow_card",
            "Bags Checked": "number_of_bages_checked",
            "Meal Type": "meal_type",
        }
    )


def transform_data(preproc_data: pl.LazyFrame) -> pl.LazyFrame:
    """Transform the preprocessed data.

    Parameters
    ----------
    preproc_data : pl.LazyFrame
        Preprocessed data as returned from preprocess_data.

    Returns
    -------
    pl.LazyFrame
        Transformed data.
    """
    return preproc_data.with_columns(
        pl.col("flight_details")
        .str.extract_groups(r"(.+)//(.+)//(.+)-(.+)//(.+)//(.+)")
        .struct.rename_fields(
            ["date", "flight_number", "from", "to", "class", "price"]
        ),
        pl.col("has_flow_card").cast(pl.Boolean),
    ).unnest("flight_details")
