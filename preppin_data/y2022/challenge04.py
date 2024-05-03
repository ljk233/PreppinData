"""2022: Week 4 - The Prep School - Travel Plans

The challenge suggests that it requires the data from week 1, but nothing
in the output depends on its data. We therefore discarded this direction.

Inputs
======
- __input/2022/PD 2021 WK 1 to 4 ideas - Preferences of Travel.csv

Outputs
=======
- output/2022/wk04_trip_summary.ndjson
"""

import polars as pl


MODE_OF_TRANSPORT = pl.LazyFrame(
    [
        ("Aeroplane", "Aeroplane", False),
        ("Bycycle", "Bicycle", True),
        ("Bicycle", "Bicycle", True),
        ("Car", "Car", False),
        ("Carr", "Car", False),
        ("Dad's Shoulders", "Mum's Shoulders", True),
        ("Helicopter", "Helicopter", False),
        ("Helicopeter", "Helicopter", False),
        ("Hopped", "Hopped", True),
        ("Jumped", "Jumped", True),
        ("Mum's Shoulders", "Mum's Shoulders", True),
        ("Scooter", "Scooter", True),
        ("Scootr", "Scooter", True),
        ("Scoter", "Scooter", True),
        ("Skipped", "Skipped", True),
        ("Van", "Van", False),
        ("Walk", "Walk", True),
        ("WAlk", "Walk", True),
        ("Waalk", "Walk", True),
        ("Walkk", "Walk", True),
        ("Wallk", "Walk", True),
    ],
    schema=["orig_mode_of_transport", "mode_of_transport", "is_sustainable"],
)


WEEKDAY_MAP = pl.LazyFrame(
    [
        ("M", "Monday"),
        ("Tu", "Tuesday"),
        ("W", "Wednesday"),
        ("Th", "Thursday"),
        ("F", "Friday"),
    ],
    schema=["weekday_code", "weekday"],
)


def solve(pd_input_w4_fsrc: str) -> pl.DataFrame:
    """Solve challenge 4 of Preppin' Data 2022.

    Parameters
    ----------
    pd_input_w4_fsrc : str
        Filepath of the input CSV file for Week 4.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the travel preference summary.

    Notes
    -----
    This function loads the input data, preprocesses it, and generates
    a summary of pupil travel plans.
    """

    # Preprocess the pupil travel plans data
    pupil_daily_travel_preference = preprocess_pupil_daily_travel_preference_data(
        pd_input_w4_fsrc,
        MODE_OF_TRANSPORT,
        WEEKDAY_MAP,
    )

    # Collect the output
    travel_preference_summary = pupil_daily_travel_preference.pipe(
        view_daily_travel_preference_summary
    )

    return travel_preference_summary.collect()


def preprocess_pupil_daily_travel_preference_data(
    pd_input_w4_fsrc: str,
    mode_of_transport: pl.LazyFrame,
    weekday_map: pl.LazyFrame,
) -> pl.LazyFrame:
    """Preprocess the pupil daily travel preference data.

    Parameters
    ----------
    pd_input_w4_fsrc : str
        Filepath of the input CSV file for Week 4.
    mode_of_transport : pl.LazyFrame
        LazyFrame representing the mode of study and whether or not it is
        sustainable.
    weekday_map : pl.LazyFrame
        LazyFrame representing the map between the short weekday code and
        its full name.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed pupil daily travel preference
        data.

    Notes
    -----
    This function loads the input CSV file, reshapes it, and cleans the
    data for further analysis.
    """
    return (
        load_data(pd_input_w4_fsrc)
        .pipe(reshape_pupil_daily_travel_preference_data)
        .pipe(
            clean_pupil_daily_travel_preference_data,
            mode_of_transport,
            weekday_map,
        )
    )


def load_data(pd_input_w4_fsrc: str) -> pl.LazyFrame:
    """Loads data from a CSV file.

    Parameters
    ----------
    pd_input_w4_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the loaded data.
    """

    return pl.scan_csv(pd_input_w4_fsrc)


def reshape_pupil_daily_travel_preference_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Reshape the pupil daily travel preference data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the input pupil daily travel preference data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the reshaped pupil daily travel preference data.
    """

    return data.melt(
        id_vars="Student ID",
        variable_name="weekday_code",
        value_name="orig_mode_of_transport",
    )


def clean_pupil_daily_travel_preference_data(
    reshaped_data: pl.LazyFrame,
    mode_of_transport: pl.LazyFrame,
    weekday_map: pl.LazyFrame,
) -> pl.LazyFrame:
    """Clean the reshaped pupil daily travel preference data.

    Parameters
    ----------
    reshaped_data : pl.LazyFrame
        LazyFrame representing the reshaped pupil daily travel preference
        data.
    mode_of_transport : pl.LazyFrame
        LazyFrame representing the mode of study and whether or not it is
        sustainable.
    weekday_map : pl.LazyFrame
        LazyFrame representing the map between the short weekday code and
        its full name.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the cleaned pupil daily travel preference
        data.
    """

    col_mapper = {"Student ID": "pupil_id"}

    return (
        reshaped_data.join(weekday_map, on="weekday_code")
        .join(mode_of_transport, on="orig_mode_of_transport", how="left")
        .drop("orig_mode_of_transport", "weekday_code")
        .rename(col_mapper)
    )


def view_daily_travel_preference_summary(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """View a summary of daily travel preferences of pupils.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing preprocessed pupil travel plan data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the summary of pupil travel plans.
    """

    # Aggreate the data
    num_trips_per_weekday = pre_data.pipe(aggregate_num_trips_per_weekday)

    num_trips_per_mode_weekday = pre_data.pipe(aggregate_num_trips_per_mode_weekday)

    # Expressions
    pct_trips_expr = 100 * (pl.col("num_trips") / pl.col("total_trips")).round(3)

    return (
        num_trips_per_mode_weekday.join(num_trips_per_weekday, on="weekday")
        .with_columns(pct_trips=pct_trips_expr)
        .drop("weekday_count")
    )


def aggregate_num_trips_per_weekday(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate the number of trips per weekday.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing preprocessed pupil travel plan data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the aggregated number of trips per weekday.
    """

    # Expressions
    total_trips_expr = pl.len()

    return pre_data.group_by("weekday").agg(total_trips=total_trips_expr)


def aggregate_num_trips_per_mode_weekday(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate the number of trips per mode of transport and weekday.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        LazyFrame representing preprocessed pupil travel plan data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the aggregated number of trips per mode
        of transport and weekday.
    """

    # Expressions
    num_trips_expr = pl.len()

    return pre_data.group_by("mode_of_transport", "weekday", "is_sustainable").agg(
        num_trips=num_trips_expr
    )
