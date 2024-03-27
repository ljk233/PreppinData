"""2023: Week 24 - Is it the teacher or the student? Part II

See solution output at "output/2023/wk24_low_performing_students.ndjson".
"""

import polars as pl

from .challenge23 import preprocess_student_info, preprocess_result


def solve(input_file: str) -> pl.DataFrame:
    """Solve challenge 24 of Preppin' Data 2023."""
    # Preprocess the data
    student_info = preprocess_student_info(input_file)
    result = preprocess_result(input_file)

    # Categorize student results relative to their peers
    categorize_result_expr = (
        pl.col("result")
        .qcut([0.25, 0.75], labels=["lower", "mid", "upper"])
        .over("subject")
    )

    categorized_result = result.with_columns(categorize_result_expr.alias("category"))

    # Count of categories by student
    student_category_count = (
        categorized_result.group_by("student_id", "category")
        .agg(pl.count().alias("k"))
        .pivot(values="k", columns="category", index="student_id")
    )

    # Predicate expr
    is_in_class_pred = pl.col("class").is_in(["9A", "9B"])
    is_low_performing_pred = pl.col("lower") >= 2

    return (
        categorized_result.join(student_category_count, on="student_id")
        .join(student_info, on="student_id")
        .filter(is_in_class_pred & is_low_performing_pred)
        .pivot(
            values="result_category",
            columns="subject",
            index=["student_id", "full_name", "gender", "class"],
        )
        .rename(
            {
                "English": "english_subject_rank",
                "Economics": "economics_subject_rank",
                "Psychology": "psychology_subject_rank",
            }
        )
    )
