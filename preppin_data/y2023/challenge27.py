"""2023: Week 27 - The Cost of Running the Prep School

See solution output at "output/2023/wk27_school_cost.ndjson".
"""

import polars as pl


def solve(input_fsrc: str) -> pl.DataFrame:
    """Solve challenge 27 of Preppin' Data 2023."""
    # Load the data into a data dictionary
    data_dict = pl.read_excel(input_fsrc, sheet_id=0)

    # Stack the data
    data = pl.concat(data_dict.values())

    # Complete the transformation
    return data.pipe(preprocess_data).pipe(transform_data).pipe(postprocess_data)


def preprocess_data(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the school overhead data."""
    renamer_dict = {
        "School Name": "school_name",
        "Year": "year",
        "Month": "month",
        "Name": "name",
        "Value": "value",
    }

    return data.rename(renamer_dict)


def transform_data(pre_data: pl.DataFrame) -> pl.DataFrame:
    """Transform the data."""
    # Expresions
    total_overhead_expr = pl.sum("value").over("school_name", "year", "month")
    month_year_expr = "01" + pl.col("month") + pl.col("year").cast(pl.Utf8)
    date_expr = month_year_expr.str.to_date("%d%B%Y")

    return pre_data.with_columns(
        total_overhead_expr.alias("total_monthly_cost"),
        date_expr.alias("date"),
    ).pivot(
        values="value",
        index=["school_name", "date", "total_monthly_cost"],
        columns="name",
    )


def postprocess_data(transformed_data: pl.DataFrame) -> pl.DataFrame:
    """Postprocess the transformed data by cleaning up the column names,
    sorting, and prepending a sort_order column.
    """
    renamer_dict = {
        "Electricity Cost": "electricity_cost",
        "Gas Cost": "gas_cost",
        "Maintenance Cost": "maintenance_cost",
        "Water Cost": "water_cost",
    }
    sort_by = ["school_name", "date"]

    return (
        transformed_data.rename(renamer_dict)
        .sort(sort_by)
        .with_row_index("sort_order", offset=1)
    )
