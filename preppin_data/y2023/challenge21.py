"""2023: Week 21- Prep School Grades

See solution output at "output/2023/wk21_students_with_cause_for_concern.ndjson".
"""

import polars as pl

from .. import common_expressions as cx


def solve(input_file: str) -> pl.DataFrame:
    """Solve challenge 21 of Preppin' Data 2023.

    Notes
    -----
    We use the median rather than the mean because the gradings are ordinal
    values, rather than numeric values. We also do not categorise the students
    because it provides little additional meaningful information.
    """
    # Preprocess the data
    pre_data = preprocess_data(input_file)

    # Aggregate the data using the median
    agg_data = pre_data.pivot(
        values="grade",
        index=["student_id", "first_name", "last_name", "gender", "born_on"],
        columns="year",
        aggregate_function="median",
    )

    return agg_data.filter(pl.col("2021") > pl.col("2022"))


def preprocess_data(input_file: str) -> pl.DataFrame:
    """Preprocess the input data."""
    return (
        pl.read_csv(input_file)
        .melt(
            id_vars=["student_id", "first_name", "last_name", "Gender", "D.O.B"],
            value_name="grade",
        )
        .with_columns(pl.col("variable").str.split("-"))
        .select(
            "student_id",
            "first_name",
            "last_name",
            cx.if_else("Gender", "M", "male", "female").alias("gender"),
            pl.col("D.O.B").str.to_date().alias("born_on"),
            pl.col("variable").list[0].cast(pl.Int16).alias("year"),
            pl.col("variable").list[1].alias("attribute"),
            "grade",
        )
    )
