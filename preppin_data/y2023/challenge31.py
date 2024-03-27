"""2023: Week 31 - HR Month - Filling in Missing IDs

See solution output at:

- "output/2023/wk31_employee.ndjson"
- "output/2023/wk31_monthly_snapshot.ndjson"
"""

import polars as pl


def solve(
    employee_fsrc: str, monthly_snapshot_fsrc: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Solve challenge 31 of Preppin' Data 2023."""
    # Preprocess the data
    pre_employee = preprocess_employee(employee_fsrc)
    pre_monthly_snapshot = preprocess_monthly_snapshot(monthly_snapshot_fsrc)

    # Create the map of employee_id -> guid
    employee_guid_map = create_employee_guid_map(pre_employee, pre_monthly_snapshot)

    # Fill in the missig data
    filled_employee = pre_employee.pipe(fill_missing_ids, employee_guid_map)
    filled_monthly_snapshot = pre_monthly_snapshot.pipe(
        fill_missing_ids, employee_guid_map
    )

    return filled_employee.collect(), filled_monthly_snapshot.collect()


def preprocess_employee(employee_fsrc: str) -> pl.LazyFrame:
    """Preprocess the employee data."""
    renamer_dict = {
        "date_of_birth": "born_on",
        "hire_date": "hired_on",
        "leave_date": "left_on",
    }

    return pl.scan_csv(employee_fsrc, try_parse_dates=True).rename(renamer_dict)


def preprocess_monthly_snapshot(monthly_snapshot_fsrc: str) -> pl.LazyFrame:
    """Preprocess the monthly snapshot data."""
    renamer_dict = {
        "dc_nbr": "distribution_center",
        "month_end_date": "last_day_of_month",
        "hire_date": "hired_on",
        "leave_date": "left_on",
    }

    return pl.scan_csv(monthly_snapshot_fsrc, try_parse_dates=True).rename(renamer_dict)


def create_employee_guid_map(
    pre_employee: pl.LazyFrame, pre_monthly_snapshot: pl.LazyFrame
) -> pl.LazyFrame:
    "Collect the unique combinations of employee_id and guid."
    return (
        pl.concat([pre_employee, pre_monthly_snapshot], how="diagonal")
        .select("employee_id", "guid")
        .drop_nulls()
        .unique()
    )


def fill_missing_ids(
    data: pl.LazyFrame, employee_guid_map: pl.LazyFrame
) -> pl.LazyFrame:
    """Fill missing employee_id and guid in data with values from the
    employee_guid_map.
    """

    # Expressions
    def fill_null_expr(col_name, fill_col):
        return (
            pl.when(pl.col(col_name).is_null())
            .then(fill_col)
            .otherwise(col_name)
            .alias(col_name)
        )

    return (
        data.join(employee_guid_map, on="employee_id", how="left")
        .join(employee_guid_map, on="guid", how="left")
        .with_columns(
            fill_null_expr("employee_id", "employee_id_right"),
            fill_null_expr("guid", "guid_right"),
        )
        .drop(["employee_id_right", "guid_right"])
    )
