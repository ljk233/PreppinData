"""2024: Week 1 - Prep Air's Flow Card

Inputs
======
- __input/2024/PD 2024 Wk 1 Input.csv

Outputs
=======
- output/2024/wk01_flight_details.ndjson

ChatGPT's review of the challenge
==================================

#DataReshaping #DataCleaning

The primary themes of this challenge revolve around data reshaping, and cleaning.

1. Data Reshaping:
   Reshaping the data is a fundamental aspect of this challenge. The `reshape_data`
   function splits the combined flight details into separate fields, allowing
   for easier manipulation and analysis of individual components.

2. Data Cleaning:
   The `clean_data` function handles data cleaning tasks, ensuring consistency
   and reliability in the dataset. Tasks such as type conversion and column
   renaming contribute to improving data quality and usability.

Overall, this challenge provides valuable practice in essential data preprocessing
techniques, including binning, reshaping, and cleaning, which are essential
for effective data analysis and manipulation.
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
        A LazyFrame representing the loaded flight details data.
    """
    return pl.scan_csv(pd_input_wk1_fsrc)


def reshape_flight_details_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Reshape the flight details data.

    Parameters
    ----------
    data : pl.LazyFrame
        Flight details data as a Polars LazyFrame.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame representing the reshaped flight details data.
    """
    # Expressions
    split_flight_details_expr = (
        pl.col("Flight Details")
        .str.extract_groups(r"^(.+)//(.+)//(.+)-(.+)//(.+)//(.+)$")
        .struct.rename_fields(
            ["flew_on", "flight_number", "from", "to", "seat_class", "price"],
        )
    )

    return data.with_columns(split_flight_details_expr).unnest("Flight Details")


def clean_flight_details_data(reshaped_data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the flight details data.

    Parameters
    ----------
    reshaped_data : pl.LazyFrame
        Reshaped flight details data as a Polars LazyFrame.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame representing the cleaned flight details data.
    """
    col_mapper = {
        "Flow Card?": "has_flow_card",
        "Bags Checked": "num_bags_checked",
        "Meal Type": "meal_type",
    }

    return reshaped_data.with_columns(
        pl.col("flew_on").str.to_date(),
        pl.col("price").cast(pl.Float64),
        pl.col("Flow Card?").cast(pl.Boolean),
    ).rename(col_mapper)


def map_flight_detail_seat_class(cleaned_data: pl.LazyFrame) -> pl.LazyFrame:
    """Fix the assigned seat class.

    Parameters
    ----------
    cleaned_data : pl.LazyFrame
        Cleaned flight details data as a Polars LazyFrame.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame representing the fixed flight details data, with each
        observation having the correct seat class.

    NOTES
    -----
    This function is not needed for challenge 1, but instead it was added
    to help solve challenge 2.
    """
    SEAT_CLASS_MAP = {
        "Economy": "First Class",
        "Premium Economy": "Business Class",
        "Business Class": "Premium Economy",
        "First Class": "Economy",
    }

    replace_seat_class_expr = pl.col("seat_class").replace(SEAT_CLASS_MAP)

    return cleaned_data.with_columns(replace_seat_class_expr)
