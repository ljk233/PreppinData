"""2024: Week 21 - Loyalty Points Percentages

Inputs
------
- __input/2024/Customer Spending.csv

Outputs
-------
- output/2024/wk20_customer_day_of_week_analysis.json
"""

import polars as pl


def solve(customer_spending_fsrc: str) -> pl.DataFrame:
    """Solves challenge 21 of Preppin' Data 2024.

    Parameters
    ----------
    customer_spending_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the loyalty points gender analysis.
    """

    # Load the data
    pre_data = preprocess_data(customer_spending_fsrc)

    # Collect the output
    loyalty_points_gender_analysis = view_free_byte_gender_analysis(pre_data)

    return loyalty_points_gender_analysis


def preprocess_data(customer_spending_fsrc: str) -> pl.DataFrame:
    """Load and preprocess the customer spending data.

    Parameters
    ----------
    customer_spending_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed input data.
    """

    data = load_data(customer_spending_fsrc)

    return data.pipe(clean_data)


def load_data(customer_spending_fsrc: str) -> pl.DataFrame:
    """Load the customer spending data from the input CSV file.

    Parameters
    ----------
    customer_spending_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the input customer spending data.
    """

    return pl.read_csv(customer_spending_fsrc, try_parse_dates=True)


def clean_data(data: pl.DataFrame) -> pl.DataFrame:
    """Clean the input customer spending data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the input data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the cleansed input data.
    """

    col_mapper = {
        "First Name": "first_name",
        "Last Name": "last_name",
        "Gender": "gender",
        "Receipt Number": "receipt_number",
        "Date": "purchased_on",
        "Sale Total": "sale_total",
    }

    # Expression
    is_online_expr = (
        pl.when(pl.col("Online") == "Yes").then(pl.lit(True)).otherwise(pl.lit(False))
    )

    return (
        data.with_columns(is_online=is_online_expr)
        .rename(col_mapper)
        .drop("Online", "In Person")
    )


def view_free_byte_gender_analysis(customer_spending: pl.DataFrame) -> pl.DataFrame:
    """View the loyalty points gender analysis.

    Parameters
    ----------
    customer_spending : pl.DataFrame
        DataFrame containing the preprocessed customer spending data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the proportion of each Free Byte .
    """

    # Expressions
    categorized_loyalty_point = categorized_loyalty_point_expr("sale_total")

    category_total_expr = pl.sum("count").over("category")

    prop_category_total_expr = pl.col("count") / category_total_expr

    pct_category_total_expr = 100 * prop_category_total_expr.round(4)

    return (
        customer_spending.with_columns(category=categorized_loyalty_point)
        .group_by("category", "gender")
        .agg(count=pl.len())
        .with_columns(pct=pct_category_total_expr)
    )


def view_loyalty_points(customer_spending: pl.DataFrame) -> pl.DataFrame:
    """View the loyalty points per receipt.

    Parameters
    ----------
    customer_spending : pl.DataFrame
        DataFrame containing the preprocessed customer spending data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the loyalty points per receipt.
    """

    # Expressions
    loyalty_points_category_expr = (
        pl.when(loyalty_points_expr >= 7)
        .then(pl.lit("MegaByte"))
        .when(loyalty_points_expr >= 5)
        .then(pl.lit("Byte"))
        .otherwise(pl.lit("NullByte"))
    )

    return customer_spending.select(
        "receipt_number", loyalty_points=loyalty_points_expr
    ).with_columns(loyalty_points_category=loyalty_points_category_expr)


def categorized_loyalty_point_expr(sale_total_col: str) -> pl.Expr:
    """Categorize the sale total.

    Parameters
    ----------
    sale_total_col : str
        Column name containing the sale total value.

    Returns
    -------
    pl.Expr
        Polars Expression that categorizes the number of loyalty points.
    """

    # Literals
    null_byte_customer = pl.lit("NullByte")

    byte_customer = pl.lit("Byte")

    mega_byte_customer = pl.lit("MegaByte")

    # Expressions
    loyalty_points = loyalty_points_expr(sale_total_col)

    # Predicates
    is_mega_byte_pred = loyalty_points >= 7

    is_byte_pred = loyalty_points >= 5

    return (
        pl.when(is_mega_byte_pred)
        .then(mega_byte_customer)
        .when(is_byte_pred)
        .then(byte_customer)
        .otherwise(null_byte_customer)
    )


def loyalty_points_expr(sale_total_col: str) -> pl.Expr:
    """Calculate the number of loyalty points for the total sales value.

    Parameters
    ----------
    sale_total_col : str
        Column name containing the sale total value.

    Returns
    -------
    pl.Expr
        Polars Expression that calculates the number of loyalty points.
    """

    return pl.col(sale_total_col) / 50
