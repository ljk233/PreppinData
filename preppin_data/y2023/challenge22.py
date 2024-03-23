"""2023: Week 22 - Prep School Grades

See solution output at "output/2023/wk22_students_with_cause_for_concern.ndjson".
"""

import polars as pl


def solve(input_file: str) -> pl.DataFrame:
    """Solve challenge 22 of Preppin' Data 2023."""
    # Preprocess the data
    pre_student_attendance = preprocess_student_attendance(input_file)
    pre_student_test_score = preprocess_student_test_scores(input_file)

    # Expressions
    attendance_col = pl.col("attendance_percentage")
    categorize_attendance_expr = (
        pl.when(attendance_col < 0.7)
        .then(pl.lit("Low"))
        .when(attendance_col > 0.9)
        .then(pl.lit("High"))
        .otherwise(pl.lit("Medium"))
    )

    return (
        pre_student_test_score.join(
            pre_student_attendance, on=["first_name", "last_name"]
        )
        .select(
            "student_id",
            "last_name",
            "first_name",
            attendance_col.alias("prop_attendance"),
            categorize_attendance_expr.alias("attendance_category"),
            "subject",
            "test_score",
            "approximate_test_score",
        )
        .collect()
    )


def preprocess_student_attendance(input_file: str) -> pl.LazyFrame:
    """Preprocess the student attendance data."""
    return (
        pl.read_excel(input_file, sheet_name="Attendance Figures")
        .lazy()
        .pipe(parse_student_name)
    )


def preprocess_student_test_scores(input_file: str) -> pl.LazyFrame:
    """Preprocess the student attendance data."""
    return (
        pl.read_excel(
            input_file, sheet_name="Student Test Scores", read_options={"skip_rows": 1}
        )
        .lazy()
        .with_columns(
            pl.col("subject").str.replace_many(
                ["Engish", "Math", "Sciece"], ["English", "Mathematics", "Science"]
            ),
            pl.col("test_date").str.to_date("%m/%d/%Y"),
            pl.col("test_score").round().cast(pl.Int8).alias("approximate_test_score"),
        )
        .pipe(parse_student_name)
    )


def parse_student_name(
    data: pl.LazyFrame, col_name: str = "student_name"
) -> pl.LazyFrame:
    """Parse the student name column into first name and last name.

    Parameters
    ----------
    data : pl.LazyFrame
        The Polars LazyFrame containing the data to be processed.
    col_name : str, optional
        The name of the column containing the full student names, by default
        "student_name".

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame with the student names split into first name
        and last name columns.

    Notes
    -----
    This function assumes that the student names are stored in a single
    column separated by underscore (_).
    """
    return (
        data.with_columns(pl.col(col_name).str.split("_"))
        .with_columns(
            pl.col(col_name).list[0].alias("first_name"),
            pl.col(col_name).list[1].alias("last_name"),
        )
        .drop(col_name)
    )
