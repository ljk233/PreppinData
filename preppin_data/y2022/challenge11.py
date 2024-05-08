"""2022: Week 11 - The Prep School - Filling the Blanks

Inputs
------
- __input/2022/PD Fill the Blanks challenge.csv

Outputs
-------
- output/2022/wk11_mean_lesson_attendance.ndjson

Notes
-----
The solution uses normalization to handle the missing data, rather than
imputation using forward/backward filling. This removes the need to sort
the data.
"""

import polars as pl


def solve(pd_fill_blanks_fsrc: str) -> pl.DataFrame:
    """Solve challenge 8 of Preppin' Data 2022.

    Parameters
    ----------
    pd_fill_blanks_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.DataFrame

    Notes
    -----
    This function reads the input CSV file, preprocesses the data, normalizes
    the lesson data and weekly lesson data, calculates the mean number
    of attendees per lesson, and returns a DataFrame containing this
    information.
    """

    # Load and preprocess the data
    pre_data = preprocess_data(pd_fill_blanks_fsrc)

    # Normalize the data
    lesson = pre_data.pipe(normalize_lesson_data)

    lesson_week = pre_data.pipe(normalize_lesson_week_data)

    # Collect the output
    mean_lesson_attendance = view_mean_lesson_num_attended(lesson, lesson_week)

    return mean_lesson_attendance


def preprocess_data(pd_fill_blanks_fsrc: str) -> pl.DataFrame:
    """Preprocess the input data.

    Parameters
    ----------
    pd_fill_blanks_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed input data.

    Notes
    -----
    We clean the column names by setting them to `snake_case` and fix the
    data types. The data still has missing values, which we will handle
    by normalization.
    """

    col_mapper = {
        "Weekday": "weekday_name",
        "Week": "week_num",
        "Teacher": "teacher_name",
        "Lesson Name": "lesson_name",
        "Subject": "subject_name",
        "Attendance": "num_attended",
    }

    # Expressions
    starts_at_expr = pl.col("Lesson Time").str.to_time("%H:%M")

    return (
        load_data(pd_fill_blanks_fsrc)
        .with_columns(starts_at=starts_at_expr)
        .drop("Lesson Time")
        .rename(col_mapper)
    )


def load_data(fsrc: str) -> pl.DataFrame:
    """Load data from the input Excel file.

    Parameters
    ----------
    fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the input data.
    """

    return pl.read_csv(fsrc)


def normalize_lesson_data(pre_data: pl.DataFrame) -> pl.DataFrame:
    """Normalize the lesson data.

    Parameters
    ----------
    pre_data : pl.DataFrame
        DataFrame containing the preprocessed input data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the normalized lesson data.

    Notes
    -----
    Primary key is {weekday, starts_at}
    """

    primary_key = ["weekday_name", "starts_at"]

    return (
        pre_data.melt(
            id_vars=primary_key,
            value_vars=["lesson_name", "subject_name"],
        )
        .drop_nulls()
        .unique()
        .pivot(
            values="value",
            index=primary_key,
            columns="variable",
        )
    )


def normalize_lesson_week_data(data: pl.DataFrame):
    """Normalize the weekly lesson data.

    Parameters
    ----------
    pre_data : pl.DataFrame
        DataFrame containing the preprocessed input data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the normalized weekly lesson data.

    Notes
    -----
    Primary key is {week_num, weekday, starts_at}
    """

    return data.drop("lesson_name", "subject_name")


def view_mean_lesson_num_attended(
    lesson: pl.DataFrame, lesson_week: pl.DataFrame
) -> pl.DataFrame:
    """View the mean number of attendees per lesson.

    Parameters
    ----------
    lesson : pl.DataFrame
        DataFrame containing the lesson data.
    lesson_week : pl.DataFrame
        DataFrame containing the weekly lesson data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the mean number of attendees per lesson.
    """

    # Expressions
    mean_num_attended_expr = pl.mean("num_attended").over("weekday_name", "starts_at")

    return lesson_week.join(lesson, on=["weekday_name", "starts_at"]).with_columns(
        mean_num_attended=mean_num_attended_expr
    )
