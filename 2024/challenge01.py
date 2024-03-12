"""Preppin' Data Challenge 2024 - Week 1

Task:
Clean and preprocess data related to Prep Air, focusing on the new loyalty
card called the Flow Card.

Solution Pipeline:
1. Load data from the input CSV file.
2. Preprocess the data by renaming columns, converting the "Flow Card?"
   column to boolean, and adding a primary key.
3. Clean the data by splitting the "Flight Details" field, converting date
   and price to the correct data types.
4. (Optional) Fix the "class" field by replacing specific values.
5. Export the cleaned data in NDJSON format.
"""

import polars as pl


# Parameters
# ----------
# IO path strings
# ===============
INPUT_CSV = "data/input/PD 2024 Wk 1 Input.csv"
OUTPUT_NDJSON = "data/output/flight_details.ndjson"
FIXED_OUT_NDJSON = "data/output/fixed_flight_details.ndjson"

# Preprocessing parameters
# ========================
INPUT_RENAMER_DICT = {
    "Flight Details": "flight_details",
    "Flow Card?": "has_flow_card",
    "Bags Checked": "number_of_bags_checked",
    "Meal Type": "meal_type",
}

# Cleansing parameters
# ====================
FLIGHT_DETAILS_PATTERN = r"(.+)//(.+)//(.+)-(.+)//(.+)//(.+)"
FLIGHT_DETAILS_STRUCT_FIELD_NAMES = [
    "date",
    "flight_number",
    "from",
    "to",
    "class",
    "price",
]

# Fix class praramters
# ====================
CLASS_REPLACER_DICT = {
    "Economy": "First Class",
    "First Class": "Economy",
    "Business Class": "Premium Economy",
    "Premium Economy": "Business Class",
}


def main() -> None:
    """Main function to orchestrate the data preparation process."""
    # Clean the input data
    cleansed_data = load_data(INPUT_CSV).pipe(preprocess_data).pipe(clean_data)

    # Export the clean data
    cleansed_data.collect().write_ndjson(OUTPUT_NDJSON)

    # Input data is wrong, so we also now fix the data
    cleansed_data.pipe(fix_class_field).collect().write_ndjson(FIXED_OUT_NDJSON)


def load_data(fsrc: str) -> pl.LazyFrame:
    """Load data from a CSV file into a polars LazyFrame.


    Parameters
    ----------
    fsrc : str
        File path of the CSV file.

    Returns
    -------
    LazyFrame
        Polars LazyFrame containing the loaded data.
    """
    return pl.scan_csv(fsrc)


def preprocess_data(input_data: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocess the data by renaming columns and casting the "Flow Card?"
    column to boolean and add a primary key field.

    Parameters
    ----------
    input_data : LazyFrame
        Input data in the form of a polars LazyFrame.

    Returns
    -------
    LazyFrame
        Polars LazyFrame with preprocessed data.
    """
    # Expressions
    cast_has_flow_card_expr = pl.col("has_flow_card").cast(pl.Boolean)

    return (
        input_data.with_row_index("id")
        .rename(INPUT_RENAMER_DICT)
        .with_columns(cast_has_flow_card_expr)
    )


def clean_data(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Cleanse the data by splitting the "Flight Details" field and converting
    date and price to the correct data types.

    Parameters
    ----------
    pre_data : LazyFrame
        Preprocessed data in the form of a polars LazyFrame.

    Returns
    -------
    LazyFrame
        Polars LazyFrame with cleaned data.
    """
    # Expressions
    extract_groups_expr = pl.col("flight_details").str.extract_groups(
        FLIGHT_DETAILS_PATTERN
    )
    rename_struct_fields_expr = pl.col("flight_details").struct.rename_fields(
        FLIGHT_DETAILS_STRUCT_FIELD_NAMES
    )
    cast_data_types_exprs = [
        pl.col("date").str.to_date(),
        pl.col("price").cast(pl.Float64),
    ]

    return (
        pre_data.with_columns(extract_groups_expr)
        .with_columns(rename_struct_fields_expr)
        .unnest("flight_details")
        .with_columns(cast_data_types_exprs)
    )


def fix_class_field(clean_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Fix the assignments for the "class" field based on a predefined dictionary
    of replacements.

    Parameters
    ----------
    clean_data : LazyFrame
        Cleaned data in the form of a polars LazyFrame.

    Returns
    -------
    LazyFrame
        Polars LazyFrame with fixed "class" field.
    """
    class_patterns_arr = list(CLASS_REPLACER_DICT.keys())
    class_replace_with_arr = list(CLASS_REPLACER_DICT.values())
    class_replacer_expr = pl.col("class").str.replace_many(
        class_patterns_arr, class_replace_with_arr
    )

    return clean_data.with_columns(class_replacer_expr)


if __name__ == "__main__":
    main()
