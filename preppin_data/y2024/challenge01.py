"""2024: Week 1 - Prep Air's Flow Card

Inputs
------
- __input/2024/PD 2024 Wk 1 Input.csv

Outputs
-------
- output/2024/wk01_flight_details.ndjson
"""

import polars as pl


def solve(pd_input_wk1_fsrc: str) -> pl.DataFrame:
    """Solve challenge 1 of Preppin' Data 2024.

    Parameters
    ----------
    pd_input_wk1_fsrc : str
        Filepath of the input CSV file for Week 1.

    Returns
    -------
    pl.DataFrame
        Processed data as a Polars DataFrame.

    Notes
    -----
    The function preprocesses the input data.
    """

    return preprocess_flight_detail_data(pd_input_wk1_fsrc).collect()


def preprocess_flight_detail_data(pd_input_wk1_fsrc: str) -> pl.LazyFrame:
    """Preprocess the flight details data.

    Parameters
    ----------
    pd_input_wk1_fsrc : str
        Filepath of the input CSV file for Week 1.

    Returns
    -------
    pl.LazyFrame
        A LazaFrame representing the preprocessed flight details data.

    Notes
    -----
    The function loads, reshapes, and cleans the flight details data.

    Primary key is {flight_detail_id} (this is a calculated field.)
    """

    return (
        load_flight_detail_data(pd_input_wk1_fsrc)
        .pipe(reshape_flight_details_data)
        .pipe(clean_flight_details_data)
        .with_row_index("flight_detail_id", offset=1)
    )


def load_flight_detail_data(pd_input_wk1_fsrc: str) -> pl.LazyFrame:
    """Load data from the input CSV file containing flight details data.

    Parameters
    ----------
    pd_input_wk1_fsrc : str
        Filepath of the input CSV file for Week 1.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the input flight details data.
    """

    return pl.scan_csv(pd_input_wk1_fsrc)


def reshape_flight_details_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Reshape the flight details data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the input flight details data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the reshaped flight details data.
    """

    # Expressions
    flight_detail_patt = r"^(.+)//(.+)//(.+)-(.+)//(.+)//(.+)$"

    flight_details_fields = [
        "flew_on",
        "flight_number",
        "from",
        "to",
        "seat_class",
        "price",
    ]

    split_flight_details_expr = (
        pl.col("Flight Details")
        .str.extract_groups(flight_detail_patt)
        .struct.rename_fields(flight_details_fields)
    )

    return data.with_columns(split_flight_details_expr).unnest("Flight Details")


def clean_flight_details_data(reshaped_data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the flight details data.

    Parameters
    ----------
    reshaped_data : pl.LazyFrame
        LazyFrame representing the reshaped flight details data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the cleansed flight details data.
    """

    col_mapper = {
        "Flow Card?": "has_flow_card",
        "Bags Checked": "num_bags_checked",
        "Meal Type": "meal_type",
    }

    # Expressions
    flew_on_expr = pl.col("flew_on").str.to_date()

    price_expr = pl.col("price").cast(pl.Float64)

    has_flow_card_expr = pl.col("Flow Card?").cast(pl.Boolean)

    return reshaped_data.with_columns(
        flew_on_expr,
        price_expr,
        has_flow_card_expr,
    ).rename(col_mapper)


def preprocess_fixed_flight_detail_data(pd_input_wk1_fsrc: str) -> pl.LazyFrame:
    """Preprocess the flight details data and map the incorrectly assigned
    seat classes.

    Parameters
    ----------
    pd_input_w1_fsrc : str
        Filepath of the input CSV file for Week 1.

    Returns
    -------
    pl.DataFrame
        LazyFrame representing the preprocessed flight details data with
        the correct assigned seat classes.

    Notes
    -----
    This function is not needed for challenge 1, but instead it was added
    to help solve challenge 2. It:
    - Preprocesses the input data.
    - Maps the incorrectly assigned seat classes to the correct seat
    class.

    Primary key is {flight_detail_id} (this is a calculated field.)
    """

    return preprocess_flight_detail_data(pd_input_wk1_fsrc).pipe(
        map_flight_detail_seat_class
    )


def map_flight_detail_seat_class(cleaned_data: pl.LazyFrame) -> pl.LazyFrame:
    """Fix the assigned seat class.

    Parameters
    ----------
    pl.LazyFrame
        LazyFrame representing the cleansed flight details data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the fixed flight details data, with each
        observation having the correct seat class.
    """

    seat_class_map = {
        "Economy": "First Class",
        "Premium Economy": "Business Class",
        "Business Class": "Premium Economy",
        "First Class": "Economy",
    }

    replace_seat_class_expr = pl.col("seat_class").replace(seat_class_map)

    return cleaned_data.with_columns(replace_seat_class_expr)
