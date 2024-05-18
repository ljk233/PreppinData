"""2024: Week 2 - Average Price Analysis

Inputs
------
- __input/2024/PD 2024 Wk 1 Input.csv

Outputs
-------
- output/2024/wk02_flight_details.ndjson
"""

import polars as pl

from .challenge01 import preprocess_fixed_flight_detail_data


def solve(pd_input_wk1_fsrc: str) -> pl.DataFrame:
    """Solve challenge 2 of Preppin' Data 2024.

    Parameters
    ----------
    pd_input_wk1_fsrc : str
        Filepath of the input CSV file for Week 1.

    Returns
    -------
    pl.DataFrame
    """

    # Preprocess the (fixed) flight details data
    pre_data = preprocess_fixed_flight_detail_data(pd_input_wk1_fsrc)

    # Collect the output
    quarterly_ticket_price_analysis = pre_data.pipe(
        view_quarterly_ticket_price_analysis
    )

    # Aggregate the price data
    return quarterly_ticket_price_analysis.collect()


def view_quarterly_ticket_price_analysis(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """View the aggregated quarterly ticket price, stratified by seat class
    and flow card levels.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazaFrame representing the preprocessed flight details data.

    Returns
    -------
    pl.LazyFrame
        LazaFrame representing the quarterly ticket price analysis.
    """

    calendar_quarter_expr = pl.col("flew_on").dt.quarter()

    return pre_data.group_by(
        "seat_class",
        "has_flow_card",
        calendar_quarter=calendar_quarter_expr,
    ).agg(
        minimum_price=pl.min("price"),
        median_price=pl.median("price"),
        mean_price=pl.mean("price"),
        maximum_price=pl.max("price"),
    )
