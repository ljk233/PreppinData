"""2023: Week 25 - Prep School 2023 Admissions - Part 1

See solution output at "output/2023/wk25_student_admission_1.ndjson").
"""

import polars as pl

from .. import common_expressions as cx


GRADE_MAP = pl.LazyFrame(
    [
        ("A", 1, 50),
        ("B", 2, 40),
        ("C", 3, 30),
        ("D", 4, 20),
        ("E", 5, 10),
        ("F", 6, 0),
    ],
    schema=["alpha_grade", "num_grade", "grade_score"],
).with_row_index("grade_id", offset=1_000_001)


def solve(
    east_students_fsrc: str, west_students_fsrc: str, school_lookup_fsrc: str
) -> pl.DataFrame:
    """Solve challenge 25 of Preppin' Data 2023."""
    # Preprocess the data
    pre_east_student = preprocess_student(
        east_students_fsrc, "%A, %d %B, %Y", GRADE_MAP, "alpha_grade"
    )
    pre_west_student = preprocess_student(
        west_students_fsrc, "%d/%m/%Y", GRADE_MAP, "num_grade"
    )
    pre_school_lookup = preprocess_school_lookup(school_lookup_fsrc)

    # Complete the transformation
    return (
        pl.concat([pre_east_student, pre_west_student], how="diagonal")
        .join(GRADE_MAP, on="grade_id")
        .with_columns(
            pl.col("student_id").str.extract(r"(\d+)").cast(pl.Int64),
            pl.col("student_id").str.extract(r"([A-Za-z]+)").alias("school_region"),
            pl.col("subject").str.to_lowercase(),
            (
                (pl.col("first_name") + " " + pl.col("last_name")).str.to_uppercase()
            ).alias("full_name"),
            pl.sum("grade_score").over("student_id").alias("total_grade_score"),
        )
        .join(pre_school_lookup, on="student_id")
        .collect()
        .pivot(
            values="alpha_grade",
            index=[
                "student_id",
                "full_name",
                "born_on",
                "school_region",
                "school_name",
                "total_grade_score",
            ],
            columns="subject",
        )
    )


def preprocess_student(
    fsrc: str, date_format: str, grade: pl.LazyFrame, grade_col: str
) -> pl.LazyFrame:
    """Preprocess the eastern student data."""
    return (
        pl.scan_csv(fsrc)
        .pipe(clean_student, date_format)
        .pipe(harmonize_student, grade, grade_col)
    )


def clean_student(data: pl.LazyFrame, date_format: str) -> pl.LazyFrame:
    """Clean the student data."""
    parse_date_expr = cx.parse_date("Date of Birth", date_format)
    renamer_dict = {
        "Unique Row ID": "id",
        "Student ID": "student_id",
        "Last Name": "last_name",
        "First Name": "first_name",
        "Date of Birth": "born_on",
        "Subject": "subject",
        "Grade": "grade",
    }

    return data.with_columns(parse_date_expr).rename(renamer_dict)


def harmonize_student(
    data: pl.LazyFrame, grade: pl.LazyFrame, grade_col: str
) -> pl.LazyFrame:
    """Harmonize the data so it is able to be merged."""
    to_upper_exprs = [
        pl.col(col_name).str.to_uppercase() for col_name in ["first_name", "last_name"]
    ]

    return (
        data.join(
            grade.select("grade_id", grade_col), left_on="grade", right_on=grade_col
        )
        .with_columns(to_upper_exprs)
        .drop("grade")
    )


def preprocess_school_lookup(fsrc: str) -> pl.LazyFrame:
    """Preprocess the school lookup data."""
    renamer_dict = {
        "Student ID": "student_id",
        "School Name": "school_name",
        "School ID": "school_id",
    }

    return (
        pl.scan_csv(fsrc)
        .rename(renamer_dict)
        .with_columns(
            pl.col("school_name").str.replace_many(
                ["St Marys", "Viliers Hill"], ["St. Mary's", "Villiers Hill"]
            )
        )
    )
