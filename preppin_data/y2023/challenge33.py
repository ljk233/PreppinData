"""2023: Week 33 - HR Month - Combinations

See solution output at:

- "output/2023/wk33_employee_age.ndjson"
- "output/2023/wk33_monthly_snapshot_with_age_range.ndjson"
"""

import polars as pl

from . import challenge32
from .. import common_expressions as cx


def solve(
    employee_fsrc: str, monthly_snapshot_fsrc: str, generation_fsrc: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Solves challenge 33 of Preppin' Data 2023.

    Parameters
    ----------
    employee_fsrc : str
        Filepath of the employee data source.
    monthly_snapshot_fsrc : str
        Filepath of the monthly snapshot data source.
    generation_fsrc : str
        Filepath of the generation data source.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        A tuple containing two polars DataFrames:
            1. Monthly snapshot with tenure information added.
            2. Aggregated demographic data based on the monthly snapshot.
    """
    # Load the solution from challenge 31
    employee_generation, monthly_snapshot_age_range = challenge32.solve(
        employee_fsrc, monthly_snapshot_fsrc, generation_fsrc
    )

    # Transfrom the data
    monthly_snapshot_with_tenure = monthly_snapshot_age_range.pipe(append_tenure)

    # Summarise the demgraphic data
    aggregate_demographics = monthly_snapshot_with_tenure.pipe(
        aggregate_data, employee_generation
    )

    return monthly_snapshot_with_tenure, aggregate_demographics


def append_tenure(monthly_snapshot: pl.DataFrame) -> pl.DataFrame:
    """Transforms the monthly snapshot data to include tenure information.

    Parameters
    ----------
    monthly_snapshot : pl.DataFrame
        Array containing monthly snapshot data.

    Returns
    -------
    pl.DataFrame
        Array with additional columns 'selected_date', 'tenure_years', and
        'tenure_months' added.
    """
    # Expressions
    # Select last_day_of_month or left_on, whichever is earliest
    selected_date_expr = (
        pl.when(pl.col("left_on").is_null())
        .then("last_day_of_month")
        .when(pl.col("last_day_of_month") <= pl.col("left_on"))
        .then("last_day_of_month")
        .otherwise("left_on")
    )

    # Calculate the tenure in years and months
    tenure_years_expr = (
        cx.approx_years_between("hired_on", "selected_date").floor().cast(pl.Int64)
    )

    tenure_months_expr = (
        cx.approx_months_between("hired_on", "selected_date").floor().cast(pl.Int64)
    )

    return monthly_snapshot.with_columns(
        selected_date_expr.alias("selected_date")
    ).with_columns(
        tenure_years_expr.alias("tenure_years"),
        tenure_months_expr.alias("tenure_months"),
    )


def aggregate_data(
    monthly_snapshot: pl.DataFrame,
    employee_generation: pl.DataFrame,
) -> pl.DataFrame:
    """Summarizes the demographics of the monthly snapshot.

    Parameters
    ----------
    monthly_snapshot : np.ndarray
        Array containing monthly snapshot data.
    employee_generation : np.ndarray
        Array containing employee generation data.
    distribution_reporting_month : np.ndarray
        Array containing distribution center reporting month data.

    Returns
    -------
    np.ndarray
        Array summarizing the demographic data with columns:
            'distribution_center', 'last_day_of_month', 'demographic_type',
            'demographic_detail', and 'number_of_employees'.
    """
    # Observed aggregate the data
    demographic_summary = (
        monthly_snapshot.join(employee_generation, on="employee_id")
        .melt(
            id_vars=["distribution_center", "last_day_of_month"],
            value_vars=[
                "nationality",
                "gender",
                "employee_generation_descr",
                "age_range_descr",
                "tenure_years",
            ],
            variable_name="demographic_type",
            value_name="demographic_detail",
        )
        .group_by(pl.all())
        .agg(pl.count().alias("number_of_employees"))
    )

    # Expected aggegate data
    distribution_center = demographic_summary.select(
        "distribution_center", "last_day_of_month"
    ).unique()

    demographic = demographic_summary.select(
        "demographic_type", "demographic_detail"
    ).unique()

    expected_demographic_summary = distribution_center.join(demographic, how="cross")

    return (
        pl.concat([demographic_summary, expected_demographic_summary], how="diagonal")
        .unique(
            [
                "distribution_center",
                "last_day_of_month",
                "demographic_type",
                "demographic_detail",
            ]
        )
        .with_columns(pl.col("number_of_employees").fill_null(0))
        .sort(
            "distribution_center",
            "last_day_of_month",
            "demographic_type",
            "demographic_detail",
        )
    )
