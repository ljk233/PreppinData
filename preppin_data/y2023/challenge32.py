"""2023: Week 32 - HR Month - Reshaping Generations

See solution output at:

- "output/2023/wk32_employee_age.ndjson"
- "output/2023/wk32_monthly_snapshot_with_age_range.ndjson"
"""

import polars as pl

from . import challenge31
from .. import common_expressions as cx


MIN_START_YEAR = 0
MAX_END_YEAR = 9_999

FIVE_YEAR_AGE_RANGE = pl.DataFrame(
    [
        (0, 19, "Under 20 years"),
        (20, 24, "20-24 years"),
        (25, 29, "25-29 years"),
        (30, 34, "30-34 years"),
        (35, 39, "35-39 years"),
        (40, 44, "40-44 years"),
        (45, 49, "45-49 years"),
        (50, 54, "50-54 years"),
        (55, 59, "55-59 years"),
        (60, 64, "60-64 years"),
        (65, 69, "65-69 years"),
        (70, 200, "70+ years"),
    ],
    schema={"min_age": pl.Int64, "max_age": pl.Int64, "description": pl.Utf8},
)


def solve(
    employee_fsrc: str, monthly_snapshot_fsrc: str, generation_fsrc: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Solve challenge 32 of Preppin' Data 2023."""
    # Load the solution from challenge 31
    employee, monthly_snapshot = challenge31.solve(employee_fsrc, monthly_snapshot_fsrc)

    # Preprocess the generations data
    pre_generation = preprocess_generation(
        generation_fsrc, MIN_START_YEAR, MAX_END_YEAR
    )

    # Transform the data
    employee_generation = employee.pipe(transform_employee, pre_generation)

    monthly_snapshot_with_employee_age_range = monthly_snapshot.pipe(
        transform_monthly_snapshot, employee, FIVE_YEAR_AGE_RANGE
    )

    return employee_generation, monthly_snapshot_with_employee_age_range


def preprocess_generation(
    generation_fsrc: str, min_start_year: int, max_end_year: int
) -> pl.DataFrame:
    """Preprocess the generations data by adding a description column for
    each generation, and filling in the missing start and end years.

    Notes
    -----
    Column start_year, end_year are filled with the min_year, max_year
    """
    # Expressions
    generation_expr = pl.col("generation")
    start_year_expr = pl.col("start_year")
    end_year_expr = pl.col("end_year")
    start_year_str_expr = start_year_expr.cast(pl.Utf8)
    end_year_str_expr = end_year_expr.cast(pl.Utf8)

    # Case-select generation description
    case_select_generation_descr_expr = (
        pl.when(start_year_expr.is_null())
        .then(pl.lit("(Born in or before ") + end_year_str_expr + ")")
        .when(end_year_expr.is_null())
        .then(pl.lit("(Born in or after ") + start_year_str_expr + ")")
        .otherwise(
            generation_expr + " (" + start_year_str_expr + "-" + end_year_str_expr + ")"
        )
    )

    return (
        pl.read_csv(generation_fsrc)
        .with_columns(case_select_generation_descr_expr.alias("generation_descr"))
        .with_columns(
            pl.col("start_year").fill_null(min_start_year),
            pl.col("end_year").fill_null(max_end_year),
        )
    )


def transform_employee(
    employee: pl.DataFrame, pre_generation: pl.DataFrame
) -> pl.DataFrame:
    """Transform the employee data by appending the employee's generation."""
    # Expressions
    filter_generation_pred_expr = pl.col("born_on").is_null() | pl.col(
        "born_on"
    ).dt.year().is_between("start_year", "end_year")

    return (
        employee.join(pre_generation, how="cross")
        .filter(filter_generation_pred_expr)
        .with_columns(switch_null_born_on("generation_descr"))
        .unique("employee_id")
        .drop("generation", "source", "start_year", "end_year", "generation_descr")
    )


def switch_null_born_on(target_col_name: str, born_on_col_name: str = "born_on"):
    """Return an expression that switches the selected generation with
    'Not provided' is born_on is missing.
    """
    return (
        pl.when(pl.col(born_on_col_name).is_null())
        .then(pl.lit("Not provided"))
        .otherwise(target_col_name)
    )


def transform_monthly_snapshot(
    monthly_snapshot: pl.DataFrame, employee: pl.DataFrame, age_range: pl.DataFrame
) -> pl.DataFrame:
    """Transform the monthly snapshot data by appending the employee's
    age range relative to the last day of the month.
    """
    # Expressions
    approx_age_expr = (
        cx.approx_years_between("born_on", "last_day_of_month").floor().cast(pl.Int64)
    )

    filter_generations_pred_expr = pl.col("born_on").is_null() | pl.col(
        "approx_employee_age"
    ).is_between("min_age", "max_age")

    return (
        monthly_snapshot.join(employee, on="employee_id")
        .join(age_range, how="cross")
        .with_columns(approx_age_expr.alias("approx_employee_age"))
        .filter(filter_generations_pred_expr)
        .with_columns(switch_null_born_on("description"))
        .unique(["last_day_of_month", "employee_id"])
        .select(
            "distribution_center",
            "last_day_of_month",
            "employee_id",
            "guid",
            "hired_on",
            "left_on",
            pl.col("description").alias("age_range_descr"),
        )
    )
