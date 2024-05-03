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

    # Collect the output
    total_grade_points = aggregate_total_grade_points(pre_data, GRADE_POINT)

    gpa_per_grade = pre_data.pipe(aggregate_grade_point_average_per_grade, GRADE_POINT)

    return total_grade_points.collect(), gpa_per_grade.collect()


def aggregate_total_grade_points(
    pre_data: pl.LazyFrame, grade_point_map: pl.LazyFrame
) -> pl.LazyFrame:
    """Aggregate total grade points per pupil.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing preprocessed pupil grade data.
    grade_point_map : pl.LazyFrame
        LazyFrame representing the map between an alphanumeric grades to
        its grade point value.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the total grade points per pupil.
    """

    # Collect the data
    pupil_subject_grade_point = view_pupil_subject_grade_point(
        pre_data, grade_point_map
    )

    # Expressions
    total_grade_points_expr = pl.sum("grade_point")

    return pupil_subject_grade_point.group_by("pupil_id").agg(
        total_grade_points=total_grade_points_expr
    )


def aggregate_grade_point_average_per_grade(
    pre_data: pl.LazyFrame, grade_point_map: pl.LazyFrame
) -> pl.LazyFrame:
    """Aggregate average grade point per grade.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing preprocessed pupil grade data.
    grade_point_map : pl.LazyFrame
        LazyFrame representing the map between an alphanumeric grades to
        its grade point value.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the average grade point per grade.
    """

    # Collect the data
    pupil_subject_grade_point = view_pupil_subject_grade_point(
        pre_data, grade_point_map
    )

    total_grade_points = aggregate_total_grade_points(pre_data, GRADE_POINT)

    # Expressions
    grade_point_average_expr = pl.mean("total_grade_points").round(2)

    return (
        pupil_subject_grade_point.join(total_grade_points, on="pupil_id")
        .group_by("grade")
        .agg(grade_point_average=grade_point_average_expr)
    )


def view_pupil_subject_grade_point(
    pre_data: pl.LazyFrame, grade_point_map: pl.LazyFrame
) -> pl.LazyFrame:
    """View the pupil grade points per subject.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing preprocessed pupil grade data.
    grade_point_map : pl.LazyFrame
        LazyFrame representing the map between an alphanumeric grades to
        its grade point value.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing pupil grade points per subject.
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
        grade=grade_expr,
    ).join(grade_point_map, on="grade")
