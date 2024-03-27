"""2023: Week 26 - Prep School 2023 Admissions - Part 2

See solution output at "output/2023/wk26_student_admission_2.ndjson".
"""

import polars as pl

from . import challenge25


SCHOOL_REGION_ALLOCATION = pl.LazyFrame(
    [("EAST", 75), ("WEST", 25)],
    schema=["school_region", "allocation"],
)


def solve(
    east_students_fsrc: str,
    west_students_fsrc: str,
    school_lookup_fsrc: str,
    addtional_student_info_fsrc: str,
) -> pl.DataFrame:
    """Solve challenge 26 of Preppin' Data 2023."""
    # Preprocess the data
    school_region_allocation = SCHOOL_REGION_ALLOCATION
    pre_student_info = preprocess_additional_student_info(addtional_student_info_fsrc)
    pre_student_performance = challenge25.solve(
        east_students_fsrc, west_students_fsrc, school_lookup_fsrc
    ).lazy()

    # Combine the data
    applicant = combine_data(pre_student_performance, pre_student_info)

    # Rank the students by subject and total grade score
    ranked_applicant = applicant.pipe(rank_applicant)

    # Select the applicants
    selected_applicants = ranked_applicant.pipe(select_applicant)

    # Complete the transformation
    return (
        selected_applicants.pipe(
            categorise_school_performance, school_region_allocation
        )
        .select(
            "student_id",
            "full_name",
            "born_on",
            "school_region",
            "school_name",
            "school_performance",
            "selected_subject",
            "english",
            "science",
            "maths",
            "total_grade_score",
        )
        .collect()
    )


def preprocess_additional_student_info(fsrc: str) -> pl.LazyFrame:
    """Preprocess the additional student info data."""
    renamer_dict = {
        "Date of Birth": "born_on",
        "School Name": "school_name",
        "Initials": "initials",
        "Home Address": "home_address",
        "English": "english",
        "Science": "science",
        "Maths": "maths",
        "Distance From School (Miles)": "miles_distance_to_school",
        "Subject Selection": "selected_subject",
    }

    return pl.scan_csv(fsrc, try_parse_dates=True).rename(renamer_dict)


def combine_data(pre_student_performance, pre_student_info) -> pl.LazyFrame:
    """"""
    # Expressions
    return (
        pre_student_performance.with_columns(
            pl.col("full_name").str.split(" ").alias("full_name_ls")
        )
        .with_columns(
            pl.col("full_name_ls").list.get(0).str.slice(0, 1).alias("first_initial"),
            pl.col("full_name_ls").list.get(1).str.slice(0, 1).alias("last_initial"),
        )
        .with_columns(
            (pl.col("first_initial") + pl.col("last_initial")).alias("initials")
        )
        .join(
            pre_student_info,
            on=["born_on", "initials", "school_name", "english", "maths", "science"],
            how="left",
        )
        .select(
            "student_id",
            "full_name",
            "born_on",
            "miles_distance_to_school",
            "school_region",
            "school_name",
            "selected_subject",
            "total_grade_score",
            "english",
            "science",
            "maths",
        )
    )


def rank_applicant(applicant: pl.LazyFrame) -> pl.LazyFrame:
    """Rank the student applicants by school region and selected subject."""
    # Parameters
    sort_by = [
        "school_region",
        "selected_subject",
        "miles_distance_to_school",
        "total_grade_score",
    ]
    descending_by = [
        False,
        False,
        True,
        False,
    ]

    # Expressions
    rank_applicants_expr = (
        pl.col("total_grade_score")
        .rank("ordinal")
        .over("school_region", "selected_subject", mapping_strategy="explode")
    )

    return applicant.sort(sort_by, descending=descending_by).with_columns(
        rank_applicants_expr.alias("rank")
    )


def select_applicant(ranked_applicant: pl.LazyFrame) -> pl.LazyFrame:
    """Select the applicants based on the given criteria in the problem."""

    def min_rank_pred_expr(min_rank):
        rank_expr = pl.col("rank")
        over_cols = "school_region", "selected_subject"

        return rank_expr > (pl.max("rank").over(over_cols) - min_rank)

    # Predicate expressions
    is_east_student = pl.col("school_region") == "EAST"

    return ranked_applicant.filter(
        (is_east_student & min_rank_pred_expr(15))
        | (is_east_student.not_() & min_rank_pred_expr(5))
    )


def categorise_school_performance(
    selected_applicant: pl.LazyFrame, school_region_allocation: pl.LazyFrame
) -> pl.LazyFrame:
    """Categoise the school performance based on the proportion of applicants
    accepted to study.
    """
    # Expresstions
    prop_accepted_expr = pl.count().over("school_name") / pl.col("allocation")

    is_high_performing_school_pred = pl.col("prop_accepted") == pl.max(
        "prop_accepted"
    ).over("school_region")

    is_low_performing_school = pl.col("prop_accepted") == pl.min("prop_accepted").over(
        "school_region"
    )

    case_select_school_performance_expr = (
        pl.when(is_high_performing_school_pred)
        .then(pl.lit("high"))
        .when(is_low_performing_school)
        .then(pl.lit("low"))
        .otherwise(pl.lit("middle"))
    )

    return (
        selected_applicant.join(school_region_allocation, on=["school_region"])
        .with_columns(prop_accepted_expr.alias("prop_accepted"))
        .with_columns(case_select_school_performance_expr.alias("school_performance"))
    )
