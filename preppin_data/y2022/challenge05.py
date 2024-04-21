"""2022: Week 5 The Prep School - Setting Grades

There are small differences in the assigned grades (and thus the GPA) because
the `qcut` function is not equivalent to Tableau's Tiling function.

We also deviate from the required output, instead returning the pupil's
total grade points and the GPA per grade.

Inputs
======
- __input/2022/PD 2022 WK 3 Grades.csv

Outputs
=======
- output/2022/wk05_total_grade_points.ndjson
- output/2022/wk05_gpa_per_grade.ndjson

ChatGPT's review of the challenge
=================================

#DataBinning #DataAggregation

The main themes of this challenge are data binning and data aggregation.

1. Data Binning:
   A significant part of the challenge involves binning the students based
   on their grades using the `qcut` function. `qcut` is a function in Pandas and
   Polars that bins continuous data into discrete intervals. In this challenge,
   it's used to assign letter grades to students based on their score distribution.
   However, it's noted that `qcut` may not produce exactly equivalent results
   to Tableau's Tiling function, leading to slight differences in assigned
   grades.

2. Data Aggregation:
   The final part of the challenge focuses on aggregating the binned data
   to generate a summary of pupil grade information. Aggregation involves
   calculating metrics such as the total grade points per pupil and the
   average grade point per grade. This aggregation allows for insights into
   pupil performance and overall grade distribution.

Overall, the challenge provides a comprehensive exercise in data binning,
reshaping, and aggregation, essential skills for any data analysis or data
science task.
"""

import polars as pl

from . import challenge03


GRADE_POINT = pl.LazyFrame(
    [
        ("A", 10),
        ("B", 8),
        ("C", 6),
        ("D", 4),
        ("E", 2),
        ("F", 1),
    ],
    schema=["grade", "grade_point"],
)


def solve(pd_input_w3_fsrc: str) -> tuple[pl.DataFrame]:
    """Solve challenge 5 of Preppin' Data 2022.

    Parameters
    ----------
    pd_input_w3_fsrc : str
        Filepath of the input CSV file for Week 3.

    Returns
    -------
    tuple[pl.DataFrame]
        1. A DataFrame containing pupil's total grade points
        2. A DataFrame containing the aggregated GPA per grade.

    Notes
    -----
    This function preprocesses pupil grade data, aggregates total grade
    points per pupil, and calculates the average grade point per grade.
    """
    # Preprocess the pupil grade data
    pre_data = challenge03.preprocess_pupil_grade_data(pd_input_w3_fsrc)

    # Output 1
    total_grade_points = pre_data.pipe(aggregate_total_grade_points)

    # Output 2
    gpa_per_grade = pre_data.pipe(aggregate_grade_point_average_per_grade)

    return total_grade_points.collect(), gpa_per_grade.collect()


def aggregate_grade_point_average_per_grade(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate average grade point per grade.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        DataFrame containing preprocessed pupil grade data.

    Returns
    -------
    pl.LazyFrame
        DataFrame containing average grade point per grade.
    """
    pupil_grade = pre_data.pipe(view_pupil_grade)

    total_grade_points = pre_data.pipe(aggregate_total_grade_points)

    return (
        pupil_grade.join(total_grade_points, on="pupil_id")
        .group_by("grade")
        .agg(pl.mean("total_grade_points").round(2).alias("grade_point_average"))
    )


def aggregate_total_grade_points(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate total grade points per pupil.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        DataFrame containing preprocessed pupil grade data.

    Returns
    -------
    pl.LazyFrame
        DataFrame containing total grade points per pupil.
    """
    pupil_grade = pre_data.pipe(view_pupil_grade)

    return pupil_grade.group_by("pupil_id").agg(
        pl.sum("grade_point").alias("total_grade_points")
    )


def view_pupil_grade(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """View pupil grades.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        DataFrame containing preprocessed pupil grade data.

    Returns
    -------
    pl.LazyFrame
        DataFrame containing pupil grades.
    """
    # Expressions
    grade_expr = (
        pl.col("score")
        .qcut(6, labels=["F", "E", "D", "C", "B", "A"])
        .over("subject_name")
        .cast(pl.Utf8)
    )

    return pre_data.select(
        "pupil_id",
        "subject_name",
        grade_expr.alias("grade"),
    ).join(GRADE_POINT, on="grade")
