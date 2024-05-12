"""2022: Week 13 - Pareto Parameters

Inputs
------
- __input/2022/Pareto Input.csv

Outputs
-------
- output/2022/wk13_running_prop_sales_analysis.ndjson
"""

import polars as pl


def solve(pareto_fsrc: str, prop: float = 0.8) -> pl.DataFrame:
    """Solve challenge 13 of Preppin' Data 2022.

    Parameters
    ----------
    pareto_fsrc : str
        Filepath of the input CSV file.
    prop : float
        Proportion of sales to filter the data to. Default is 0.8.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the customer whose total sales together comprise
        the given proportion of sales.
    """

    # Load and preprocess the data
    pre_data = preprocess_data(pareto_fsrc)

    # Collect the output
    running_prop_sales = pre_data.pipe(view_running_prop_sales, prop)

    return running_prop_sales.collect()


def preprocess_data(pareto_fsrc: str) -> pl.LazyFrame:
    """Preprocess the input data.

    Parameters
    ----------
    pareto_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed input data.
    """

    return load_data(pareto_fsrc).pipe(clean_data)


def load_data(fsrc: str) -> pl.LazyFrame:
    """Load data from the input Excel file.

    Parameters
    ----------
    fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the input data.
    """

    return pl.scan_csv(fsrc)


def clean_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the input data.

    Parameters
    ----------
    data : pl.DataFrame
        LazyFrame representing the input data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed data.
    """

    col_mapper = {
        "Customer ID": "customer_id",
        "First Name": "first_name",
        "Surname": "last_name",
        "Order ID": "order_id",
        "Sales": "sales",
    }

    return data.rename(col_mapper)


def view_running_prop_sales(pre_data: pl.LazyFrame, prop: float = 1.0) -> pl.LazyFrame:
    """View the running proportion of sales up to and including the given
    proportion.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing the preprocessed data.
    prop : float
        Proportion of sales to filter the data to.
        Default is 1.0.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the customer whose total sales together
        comprise the given proportion of sales.
    """

    customer_prop_sales = pre_data.pipe(view_customer_prop_sales)

    # Expressions
    running_prop_sales_expr = pl.cum_sum("prop_customer_sales")

    # Predicates
    is_running_prop_leq_prop_pred = pl.col("running_prop_sales") <= prop

    return (
        customer_prop_sales.sort("prop_customer_sales", descending=True)
        .with_columns(running_prop_sales=running_prop_sales_expr)
        .filter(is_running_prop_leq_prop_pred)
    )


def view_customer_prop_sales(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """View each customers proportion of the total sales.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing the preprocessed data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the customers proportion of total sales.
    """

    customer_total_sales = pre_data.pipe(view_customer_total_sales)

    # Expressions
    total_sales_expr = pl.sum("customer_sales")

    prop_customer_sales_expr = pl.col("customer_sales") / total_sales_expr

    return customer_total_sales.select(
        "customer_id",
        "first_name",
        "last_name",
        "customer_sales",
        prop_customer_sales=prop_customer_sales_expr,
    )


def view_customer_total_sales(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate the total sales per customer.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing the preprocessed data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the aggregate total sales by customer.
    """

    # Expressions
    customer_sales_expr = pl.sum("sales")

    return pre_data.group_by(
        "customer_id",
        "first_name",
        "last_name",
    ).agg(customer_sales=customer_sales_expr)
