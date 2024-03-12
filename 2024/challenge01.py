"""Preppin' Data Challenge 2024: Week 1 - Prep Air's Flow Card
"""

import polars as pl


# Parameters
# ----------

# IO path strings
# ===============

INPUT_CSV = "data/input/PD 2024 Wk 1 Input.csv"
OUTPUT_NDJSON = "data/output/flight_details.ndjson"
FIXED_OUT_NDJSON = "data/output/fixed_flight_details.ndjson"

# Cleansing parameters
# ====================

INPUT_RENAMER_DICT = {
    "Flow Card?": "has_flow_card",
    "Bags Checked": "number_of_bags_checked",
    "Meal Type": "meal_type",
}

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


# Functions
# ---------


def main() -> None:
    """Main function to orchestrate the data preparation process."""
    # Clean the input data
    cleansed_data = load_data(INPUT_CSV).pipe(clean_data)

    # Export the clean data
    cleansed_data.collect().write_ndjson(OUTPUT_NDJSON)

    # Input data is wrong, so we also now fix the data
    cleansed_data.pipe(fix_class_field).collect().write_ndjson(FIXED_OUT_NDJSON)


def load_data(fsrc: str) -> pl.LazyFrame:
    """Load data from a CSV file into a polars LazyFrame."""
    return pl.scan_csv(fsrc)


def clean_data(input_data: pl.LazyFrame) -> pl.LazyFrame:
    """Cleanse the data.

    Notes
    -----
    Cleaning steps:
    1. Add a primary key
    2. Split the "Flight Details" column
    3. Cast the data, price, and flow card coluns to the correct data types
    """
    # Expressions
    extract_groups_expr = (
        pl.col("Flight Details")
        .str.extract_groups(FLIGHT_DETAILS_PATTERN)
        .alias("flight_details_struct")
    )
    rename_struct_fields_expr = pl.col("flight_details_struct").struct.rename_fields(
        FLIGHT_DETAILS_STRUCT_FIELD_NAMES
    )
    cast_data_types_exprs = [
        pl.col("Flow Card?").cast(pl.Boolean),
        pl.col("date").str.to_date(),
        pl.col("price").cast(pl.Float64),
    ]

    return (
        input_data
        # Add a primary key
        .with_row_index("id")
        # Split the flight detail columns
        .with_columns(extract_groups_expr)
        .with_columns(rename_struct_fields_expr)
        .unnest("flight_details_struct")
        # Clean the data types
        .with_columns(cast_data_types_exprs)
        # Clean the column name
        .rename(INPUT_RENAMER_DICT)
        .select("id", *FLIGHT_DETAILS_STRUCT_FIELD_NAMES, *INPUT_RENAMER_DICT.values())
    )


def fix_class_field(clean_data: pl.LazyFrame) -> pl.LazyFrame:
    """Fix the assignments for the "class" columns.

    Notes
    -----
    We learned that class column contained the incorrect data in the second
    challenge.
    """
    class_patterns_arr = list(CLASS_REPLACER_DICT.keys())
    class_replace_with_arr = list(CLASS_REPLACER_DICT.values())

    # Expressions
    class_replacer_expr = pl.col("class").str.replace_many(
        class_patterns_arr, class_replace_with_arr
    )

    return clean_data.with_columns(class_replacer_expr)


if __name__ == "__main__":
    main()
