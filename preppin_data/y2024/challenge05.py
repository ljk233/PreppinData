"""2024: Week 5 - Getting the right data

We modified the challenge. There is a small element of datetime arithmetic
in the third output. This isn't particularly novel, so we ingored that
direction and instead elaborated exporting the three output datasets to
a single JSON-like string. The problem now has just two themes:

1. Use of anti-joins.
2. Storing a collection of datasets in a single JSON file.

Inputs
======
- __input/2024/Prep Air 2024 Flights.csv
- __input/2024/Prep Air Customers.csv
- __input/2024/Prep Air Ticket Sales.csv

Outputs
=======
- output/2024/wk05_booking_summary.json
"""

from datetime import date
import json
from pathlib import Path

import polars as pl


REPORTED_ON = date(2024, 1, 31)


def solve(
    flights_fsrc: str,
    customers_fsrc: str,
    ticket_sales_fsrc: str,
) -> str:
    """Solves challenge 5 of Preppin' Data 2024.

    Parameters
    ----------
    flights_fsrc : str
        Path to the CSV file containing flights data.
    customers_fsrc : str
        Path to the CSV file containing customers data.
    ticket_sales_fsrc : str
        Path to the CSV file containing ticket sales data.

    Returns
    -------
    str
        JSON-like string representing the required output data.
    """

    # Preprocess the source data
    flight = preprocess_data(flights_fsrc)

    customer = preprocess_data(customers_fsrc)

    ticket_sale = preprocess_data(ticket_sales_fsrc)

    # Collect the output
    output_data_dict = collect_output(flight, customer, ticket_sale)

    return json.dumps(output_data_dict, indent=2, default=str)


def preprocess_data(fsrc: str) -> pl.DataFrame:
    """Load and preprocesses data from the given CSV file.

    Parameters
    ----------
    fsrc : str
        Path to the source CSV file.

    Returns
    -------
    pl.LazyFrame
        DataFrame containing the preprocessed data.
    """

    schema = get_partial_schema(fsrc)

    return pl.read_csv(fsrc, new_columns=schema, try_parse_dates=True)


def get_partial_schema(fsrc: str) -> list[str]:
    """Return the partial schema for the given file.

    Parameters
    ----------
    fsrc: str
        Path to the source CSV file.

    Returns
    -------
    list[str]
        List containing column names representing the partial schema of
        the file.
    """

    # Extract the file name
    file_name = get_file_name(fsrc)

    # Match and return the schema for the file name
    match file_name:
        case "Prep Air 2024 Flights":
            return ["flies_on", "flight_number", "flies_from", "flies_to"]
        case "Prep Air Customers":
            return ["customer_id", "last_flew_on"]
        case "Prep Air Ticket Sales":
            return ["flies_on", "flight_number", "customer_id", "ticket_price"]


def get_file_name(fsrc: str) -> str:
    """Extract the file name from the give path string.

    Parameters
    ----------
    fsrc : str
        Path to the source CSV file.

    Returns
    -------
    str
        File name of the fiven path string (without the file type.)
    """

    return Path(fsrc).stem


def collect_output(
    flight: pl.LazyFrame,
    customer: pl.LazyFrame,
    ticket_sale: pl.LazyFrame,
) -> dict[str, pl.LazyFrame]:
    """Collects the output dataframes into a dictionary.

    Parameters
    ----------
    flight : pl.DataFrame
        DataFrame containing the preprocessed flight data.
    customer : pl.DataFrame
        DataFrame containing the preprocessed customer data.
    ticket_sale : pl.DataFrame
        DataFrame containing the preprocessed ticket sale data.

    Returns
    -------
    dict[str, pl.LazyFrame]
        A dictionary containing the output dataframes.
    """

    # Collect the outputs
    ticket_sale_detail = view_ticket_sale_detail(flight, ticket_sale, ticket_sale)

    unbooked_flight = view_unbooked_flights(flight, ticket_sale)

    unbooked_customer = view_unbooked_customers(customer, ticket_sale)

    return {
        "ticket_sale_detail": ticket_sale_detail.to_dicts(),
        "unbooked_flights": unbooked_flight.to_dicts(),
        "unbooked_customers": unbooked_customer.to_dicts(),
    }


def view_ticket_sale_detail(
    flight: pl.LazyFrame,
    customer: pl.LazyFrame,
    ticket_sale: pl.LazyFrame,
) -> pl.LazyFrame:
    """View the detail of ticket sales, including flight and customer
    information.

    Parameters
    ----------
    flight : pl.DataFrame
        DataFrame containing the preprocessed flight data.
    customer : pl.DataFrame
        DataFrame containing the preprocessed customer data.
    ticket_sale : pl.DataFrame
        DataFrame containing the preprocessed ticket sale data.

    Returns
    -------
    pl.LazyFrame
        DataFrame containing the ticket sale data of booked flights.
    """

    return ticket_sale.join(customer, on="customer_id").join(
        flight, on=["flight_number", "flies_on"]
    )


def view_unbooked_flights(
    flight: pl.DataFrame, ticket_sale: pl.DataFrame
) -> pl.DataFrame:
    """Views unbooked flights based on ticket sales data.

    Parameters
    ----------
    flight : pl.DataFrame
        DataFrame containing the preprocessed flight data.
    ticket_sale : pl.DataFrame
        DataFrame containing the preprocessed ticket sale data.

    Returns
    -------
    pl.DataFrame
        Dataframe containing unbooked flights.
    """

    return flight.join(
        ticket_sale,
        on=["flight_number", "flies_on"],
        how="anti",
    ).with_columns(reported_on=REPORTED_ON)


def view_unbooked_customers(
    customer: pl.LazyFrame, pre_ticket_sale: pl.LazyFrame
) -> pl.LazyFrame:
    """Views unbooked customers based on ticket sales data.

    Parameters
    ----------
    customer : pl.DataFrame
        DataFrame containing the preprocessed customer data.
    ticket_sale : pl.DataFrame
        DataFrame containing the preprocessed ticket sale data.

    Returns
    -------
    pl.DataFrame
        Dataframe containing the unbooked customers.
    """
    return customer.join(
        pre_ticket_sale,
        on="customer_id",
        how="anti",
    )
