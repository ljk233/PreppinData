"""2024: Week 6 - Staff Income Tax

Inputs
------
- __input/2024/PD 2024 Wk 6 Input.csv

Outputs
-------
- output/2024/wk06_tax_summary.ndjson
"""

import polars as pl


INCOME_TAX_BAND = pl.DataFrame(
    [
        ("basic", 12_570, 50_270, 0.2),
        ("higher", 50_270, 125_140, 0.4),
        ("additional", 125_140, None, 0.45),
    ],
    schema=["tax_band_code", "lower_threshold", "upper_threshold", "tax_rate"],
)


def solve(pd_input_wk6_fsrc: str) -> pl.DataFrame:
    """Solve challenge 6 of Preppin' Data 2024.

    Parameters
    ----------
    pd_input_wk6_fsrc : str
        Filepath of the input CSV file for Week 6.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the tax summary for each staff member.

    Notes
    -----
    This function loads the input CSV file for Week 6, preprocesses the
    monthly salary data, and then generates a summary of tax information
    for each staff member based on their monthly salaries.
    """

    # Load and preprocess the data
    pre_monthly_salary_data = preprocess_monthly_salary_data(pd_input_wk6_fsrc)

    # Collect the output
    tax_summary = view_tax_summary(pre_monthly_salary_data)

    return tax_summary


def preprocess_monthly_salary_data(pd_input_wk6_fsrc: str) -> pl.DataFrame:
    """Load and preprocess the monthly salary data.

    Parameters
    ----------
    pd_input_wk6_fsrc : str
        Filepath of the input CSV file for Week 6.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed monthly salary data.

    Notes
    -----
    The function loads, filters, reshapes, and cleans the source data.
    Primary key is {staff_id, month_num}.
    """

    data = load_monthly_salary_data(pd_input_wk6_fsrc)

    return (
        data.pipe(filter_monthly_salary_data)
        .pipe(reshape_monthly_salary_data)
        .pipe(clean_monthly_salary_data)
    )


def load_monthly_salary_data(pd_input_wk6_fsrc: str) -> pl.DataFrame:
    """Load the staff monthly salary.

    Parameters
    ----------
    pd_input_wk6_fsrc : str
        Filepath of the input CSV file for Week 6.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the input monthly salary data.
    """

    return pl.read_csv(pd_input_wk6_fsrc)


def filter_monthly_salary_data(data: pl.DataFrame) -> pl.DataFrame:
    """Filter the staff monthly salary data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the input monthly salary data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the filtered monthly salary data.

    Notes
    -----
    Data can contain one or many rows for each Staff ID. This function
    selects the latest record.
    """

    return (
        data.with_row_index("index")
        .filter(pl.col("index") == pl.max("index").over("StaffID"))
        .drop("index")
    )


def reshape_monthly_salary_data(filtered_data: pl.DataFrame) -> pl.DataFrame:
    """Reshape the filtered monthly salary data.

    Parameters
    ----------
    filtered_data : pl.DataFrame
        DataFrame containing the filtered monthly salary data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the reshaped monthly salary data.

    Notes
    -----
    Each row in the returned LazyFrame represents the salary received by
    a staff member in a particular month.
    """

    return filtered_data.melt(
        id_vars=["StaffID"],
        variable_name="month_num",
        value_name="salary",
    ).with_columns(pl.col("month_num").cast(pl.Int8))


def clean_monthly_salary_data(reshaped_data: pl.DataFrame) -> pl.DataFrame:
    """Clean the reshapend monthly salary data.

    Parameters
    ----------
    reshaped_data : pl.DataFrame
        DataFrame containing the reshaped monthly salary data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing  the cleaned monthly salary data.
    """

    col_mapper = {"StaffID": "staff_id"}

    return reshaped_data.rename(col_mapper)


def view_tax_summary(pre_data: pl.DataFrame) -> pl.DataFrame:
    """View the summary of tax information for staff members.

    Parameters
    ----------
    pre_data : pl.DataFrame
        DataFrame containing the preprocessed monthly salary data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing a tax summary for each staff member.
    """

    # Collect the data
    annual_salary = pre_data.pipe(view_annual_salary)

    # Expressions
    max_tax_rate_expr = (
        pl.when(pl.col("additional").is_not_null())
        .then(pl.lit("Additional (45%)"))
        .when(pl.col("higher").is_not_null())
        .then(pl.lit("Higher (40%)"))
        .when(pl.col("basic").is_not_null())
        .then(pl.lit("Basic (20%)"))
        .otherwise(pl.lit("No liability"))
    )

    return annual_salary.pipe(view_tax_liability, INCOME_TAX_BAND).with_columns(
        max_tax_rate=max_tax_rate_expr
    )


def view_annual_salary(pre_data: pl.DataFrame) -> pl.DataFrame:
    """View the annual salary paid by staff.

    Parameters
    ----------
    pre_data : pl.DataFrame
        DataFrame containing the preprocessed monthly salary data

    Returns
    -------
    pl.DataFrame
        DataFrame containing the annual salary earned by each staff member.
    """

    # Expressions
    annual_salary_expr = annual_salary = pl.sum("salary")

    return pre_data.group_by("staff_id").agg(annual_salary=annual_salary_expr)


def view_tax_liability(
    annual_salary: pl.DataFrame, income_tax_band: pl.DataFrame
) -> pl.DataFrame:
    """View each staff's annual tax liability at each income tax band.

    Parameters
    ----------
    annual_salary : pl.DataFrame
        DataFrame containing the annual salary for each staff member.
    income_tax_band : pl.DataFrame
        DataFrame containing the income tax bands.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the tax liability for each staff member.

    Notes
    -----
    Primary key is {staff_id}
    """

    # Scaffold the data
    scaffolded_data = annual_salary.join(income_tax_band, how="cross")

    # Expressions
    upper_salary_expr = pl.min_horizontal("annual_salary", "upper_threshold")

    threshold_salary_expr = upper_salary_expr - pl.col("lower_threshold")

    tax_paid_expr = pl.col("tax_rate") * threshold_salary_expr

    total_tax_liability_expr = pl.sum("tax_paid").over("staff_id")

    # Predicates
    is_positive_tax_liability_pred = pl.col("tax_paid") >= 0

    return (
        scaffolded_data.with_columns(tax_paid=tax_paid_expr.round(2))
        .filter(is_positive_tax_liability_pred)
        .with_columns(total_tax_liability=total_tax_liability_expr)
        .pivot(
            values="tax_paid",
            index=[
                "staff_id",
                "annual_salary",
                "total_tax_liability",
            ],
            columns="tax_band_code",
        )
    )
