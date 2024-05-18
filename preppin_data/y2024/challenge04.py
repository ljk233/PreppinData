"""2024: Week 4 - Unpopular Seats

Inputs
------
- __input/2024/PD 2024 Wk 4 Input.xlsx

Outputs
-------
- output/2024/wk04_unbooked_seats.ndjson
"""

import polars as pl


def solve(pd_input_wk4_fsrc: str) -> pl.DataFrame:
    """Solve challenge 4 of Preppin' Data 2024.

    Parameters
    ----------
    pd_input_wk4_fsrc : str
        Filepath of the input XLSX file for Week 4.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the unbooked seats.
    """

    # Load and preprocess the data
    seat_allocation = preprocess_seat_allocation_data(pd_input_wk4_fsrc)

    seat_plan = preprocess_seat_plan_data(pd_input_wk4_fsrc)

    # Collect the output
    unbooked_seats = view_unbooked_seats(seat_allocation, seat_plan)

    return unbooked_seats


def preprocess_seat_allocation_data(pd_input_wk4_fsrc: str) -> pl.DataFrame:
    """Load and preprocess the seat allocation data.

    Parameters
    ----------
    pd_input_wk4_fsrc : str
        Filepath of the input XLSX file for Week 4.

    Returns
    -------
    pl.DataFrame
        A DataFrame representing the seat allocation data.
    """

    data_ls = load_seat_allocation_data(pd_input_wk4_fsrc)

    return merge_seat_allocation_data(*data_ls).pipe(clean_seat_allocation_data)


def load_seat_allocation_data(pd_input_wk4_fsrc: str) -> list[pl.DataFrame]:
    """Load the seat allocation data from the source XLSX file.

    Parameters
    ----------
    pd_input_wk4_fsrc : str
        Filepath of the input XLSX file.

    Returns
    -------
    list[pl.DataFrame]
        A list of DataFrames containing the input seat allocation data.
    """

    data_dict = load_data(pd_input_wk4_fsrc)

    return [
        data
        for sheet_name, data in data_dict.items()
        if sheet_name in ["Flow Card", "Non_flow Card", "Non_flow Card2"]
    ]


def load_data(pd_input_wk4_fsrc: str) -> dict[str, pl.DataFrame]:
    """Load the source data from an XLSX file.

    Parameters
    ----------
    pd_input_wk4_fsrc : str
        Filepath of the input XLSX file.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary where keys are worksheet names and values are DataFrames.
        Each DataFrame contains the data loaded from a worksheet in the Excel
        file.
    """

    return pl.read_excel(pd_input_wk4_fsrc, sheet_id=0)


def merge_seat_allocation_data(*data: pl.DataFrame) -> pl.DataFrame:
    """Merge seat allocation data from multiple sources.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the seat allocation data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the merged seat allocation data.
    """

    return pl.concat(data)


def clean_seat_allocation_data(data: pl.DataFrame) -> pl.DataFrame:
    """Clean the seat allocation data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the seat allocation data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the cleaned seat allocation data.
    """

    col_mapper = {
        "CustomerID": "customer_id",
        "Seat": "seat",
        "Row": "row",
        "Class": "seat_class_code",
    }

    return data.rename(col_mapper)


def preprocess_seat_plan_data(pd_input_wk4_fsrc: str) -> pl.DataFrame:
    """Load and preprocess the seat plan data.

    Parameters
    ----------
    pd_input_wk4_fsrc : str
        Filepath of the input XLSX file for Week 4.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed seat plan data.
    """

    data = load_seat_plan_data(pd_input_wk4_fsrc)

    return data.pipe(clean_seat_plan_data)


def load_seat_plan_data(pd_input_wk4_fsrc: str) -> list[pl.DataFrame]:
    """Load the seat plan data from the source XLSX file.

    Parameters
    ----------
    pd_input_wk4_fsrc : str
        Filepath of the input XLSX file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the input seat plan data.
    """

    data_dict = load_data(pd_input_wk4_fsrc)

    return data_dict["Seat Plan"]


def clean_seat_plan_data(data: pl.DataFrame) -> pl.DataFrame:
    """Clean the seat plan data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the seat plan data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the cleansed seat plan data.
    """

    col_mapper = {
        "Seat": "seat",
        "Row": "row",
        "Class": "seat_class_code",
    }

    return data.rename(col_mapper)


def view_unbooked_seats(
    seat_allocation: pl.DataFrame, seat_plan: pl.DataFrame
) -> pl.DataFrame:
    """View unbooked seats based on seat allocation and seat plan data.

    Parameters
    ----------
    seat_allocation : pl.DataFrame
        DataFrame containing the seat allocation data.
    seat_plan : pl.DataFrame
        DataFrame containing the seat plan data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the unbooked seats.
    """

    return seat_plan.join(
        seat_allocation,
        on=["seat", "row", "seat_class_code"],
        how="anti",
    )
