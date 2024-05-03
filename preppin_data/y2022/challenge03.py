"""2022: Week 3 - The Prep School - Passing Grades

There is some confusion throughout the challenge, because the terms "grade"
and "score" are used interchangably despite them not being equivalent.
We use "score" in this challenge.

Inputs
======
- __input/2022/PD 2022 Wk 1 Input - Input.csv
- __input/2022/PD 2022 WK 3 Grades.csv

Outputs
=======
- output/2022/wk03_pupil_performance_summary.ndjson
"""

import polars as pl

from . import challenge01


def solve(pd_input_wk1_fsrc: str, pd_input_w3_fsrc: str) -> pl.DataFrame:
    """Solve challenge 3 of Preppin' Data 2022.

    Parameters
    ----------
    pd_input_wk1_fsrc : str
        Filepath of the input CSV file for Week 1.
    pd_input_w3_fsrc : str
        Filepath of the input CSV file for Week 3.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing pupil performance data.

    Notes
    -----
    This function preprocesses pupil data from Week 1 and pupil grade data
    from Week 3, joins them, and calculates pupil performance metrics such
    as the number of subjects passed and mean score.
    """

    # Preprocess and normalise the pupil data
    pre_week1_data = challenge01.preprocess_pupil_contact_data(pd_input_wk1_fsrc)

    pupil = pre_week1_data.pipe(challenge01.normalize_pupil)

    # Preprocess the pupil grade data
    pupil_grade = preprocess_pupil_grade_data(pd_input_w3_fsrc)

    # Collect the output
    pupil_performance_summary = view_pupil_performance_summary(pupil, pupil_grade)

    return pupil_performance_summary.collect()


def preprocess_pupil_grade_data(pd_input_w3_fsrc: str) -> pl.LazyFrame:
    """Preprocess the pupil grade data.

    Parameters
    ----------
    pd_input_w3_fsrc : str
        Filepath of the input CSV file for Week 3.

    Returns
    -------
    pl.LazyFrame
        Preprocessed pupil grade data.

    Notes
    -----
    This function loads the pupil grade data from Week 3, reshapes it, and
    cleans the data. Primary key is {pupil_id, subject_name}.
    """
    return (
        load_pupil_grade_data(pd_input_w3_fsrc)
        .pipe(reshape_pupil_grade_data)
        .pipe(clean_pupil_grade_data)
    )


def load_pupil_grade_data(pd_input_w3_fsrc: str) -> pl.LazyFrame:
    """Load the pupil grade data.

    Parameters
    ----------
    pd_input_w3_fsrc : str
        Filepath of the input CSV file for Week 3.

    Returns
    -------
    pl.LazyFrame
        Loaded pupil grade data.

    Notes
    -----
    This function scans the CSV file containing pupil grade data.
    """
    return pl.scan_csv(pd_input_w3_fsrc)


def reshape_pupil_grade_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Reshape the pupil grade data.

    Parameters
    ----------
    data : pl.LazyFrame
        Input data.

    Returns
    -------
    pl.LazyFrame
        Reshaped pupil grade data.

    Notes
    -----
    This function reshapes the pupil grade data from wide to long format.
    """
    return data.melt(
        id_vars="Student ID",
        variable_name="subject_name",
        value_name="score",
    )


def clean_pupil_grade_data(reshaped_data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the pupil grade data.

    Parameters
    ----------
    reshaped_data : pl.LazyFrame
        Reshaped data.

    Returns
    -------
    pl.LazyFrame
        Cleaned pupil grade data.

    Notes
    -----
    This function renames columns in the reshaped pupil grade data.
    """
    col_mapper = {"Student ID": "pupil_id"}

    return reshaped_data.rename(col_mapper)


def view_pupil_performance_summary(
    pupil: pl.LazyFrame, pupil_grade: pl.LazyFrame
) -> pl.LazyFrame:
    """View a summary of each pupil's performance.

    Parameters
    ----------
    pupil : pl.LazyFrame
        LazyFrame representing the normalized pupil data.
    pupil_score : pl.LazyFrame
        LazyFrame representing the normalized pupil data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing a summary of each pupil's performance.

    Notes
    -----
    This function calculates pupil performance metrics such as the number
    of subjects passed and the mean score.

    Primary key is {pupil_id}.
    """

    # Collect the data
    pupil_performance = view_pupil_performance(pupil, pupil_grade)

    # Expressions
    num_subjects_passed_expr = pl.sum("did_pass")

    mean_score_expr = pl.mean("score")

    return pupil_performance.group_by("pupil_id", "gender").agg(
        num_subjects_passed=num_subjects_passed_expr,
        mean_score=mean_score_expr,
    )


def view_pupil_performance(
    pupil: pl.LazyFrame, pupil_grade: pl.LazyFrame
) -> pl.LazyFrame:
    """View the pupil's performance in different subjects.

    Parameters
    ----------
    pupil : pl.LazyFrame
        LazyFrame representing the normalized pupil data.
    pupil_grade : pl.LazyFrame
        LazyFrame representing the pupil grade data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the pupil's grades in different subjects,
        and whether they passed or not.

    Notes
    -----
    Primary key is {pupil_id, subject_name}.
    """

    # Expressions
    did_pass_expr = (
        pl.when(pl.col("score") >= 75).then(pl.lit(True)).otherwise(pl.lit(False))
    )

    return pupil.join(pupil_grade, on="pupil_id").with_columns(did_pass=did_pass_expr)
