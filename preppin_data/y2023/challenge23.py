"""2023: Week 23 - Is it the teacher or the student? Part I


See solution output at "output/2023/wk23_lowest_ranked_class_by_subject.ndjson".
"""

import polars as pl


def solve(input_file: str) -> pl.DataFrame:
    """Solve challenge 23 of Preppin' Data 2023."""
    # Preprocess the data
    student_info = preprocess_student_info(input_file)
    result = preprocess_result(input_file)

    return (
        student_info.join(result, on="student_id")
        .group_by("class", "subject")
        .agg(pl.mean("result").alias("mean_result"))
        .with_columns(pl.col("mean_result").rank("max").over("subject").alias("rank"))
        .filter(pl.col("rank") == 1)
        .drop("rank")
    )


def preprocess_student_info(input_file: str) -> pl.DataFrame:
    """Preprocess the student info data."""
    return (
        pl.read_excel(input_file, sheet_name="Student Info")
        .with_columns(pl.col(pl.Utf8).str.strip_chars(" "))
        .rename(
            {
                " Student ID ": "student_id",
                " Full name         ": "full_name",
                " Gender ": "gender",
                " Class ": "class",
            }
        )
    )


def preprocess_result(input_file: str) -> pl.DataFrame:
    """Preprocess the result data."""
    return (
        pl.read_excel(input_file, sheet_name="Results")
        .rename({"Student ID": "student_id"})
        .melt(id_vars="student_id", variable_name="subject", value_name="result")
    )
