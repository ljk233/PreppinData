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

There
"""

import polars as pl


def run_pipeline(flight_details_csv_path: str) -> None:
    """Run the pipeline.

    _extended_summary_

    Parameters
    ----------
    flight_details_csv_path : str
        Path to the raw data.
    """

    raw_data = extract_data(flight_details_csv_path)
    preproc_data = raw_data.pipe(preprocess_data)
    transformed_data = preproc_data.pipe(transform_data)
    transformed_data.pipe(load_data)


def extract_data(csv_path: str, schema: dict[str, str]) -> pl.LazyFrame:
    """Extract the data into a polars LazyFrame."""
    return pl.scan_csv(fp)


def preprocess_data(raw_data: pl.LazyFrame) -> pl.LazyFrame:
    """Return the preprocessed data."""


def transform_data(preproc_data: pl.LazyFrame) -> pl.LazyFrame:
    """_summary_

    Args:
        data (pl.LazyFrame): _description_

    Returns:
        pl.LazyFrame: _description_
    """


def load_data(transformed_data: pl.LazyFrame) -> None:
    """_summary_

    Args:
        transformed_data (pl.LazyFrame): _description_
    """
