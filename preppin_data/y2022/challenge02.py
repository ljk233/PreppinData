"""2022: Week 2 - The Prep School - Birthday Cakes

Inputs
======
- __input/2022/PD 2022 Wk 1 Input - Input.csv

Outputs
=======
- output/2022/wk02_pupil_birthday_2022.ndjson
- output/2022/wk02_num_birthdays_per_month_and_weekday.ndjson
"""

import polars as pl

from . import challenge01


CALENDAR_YEAR = 2022


def solve(input_fsrc: str) -> pl.DataFrame:
    """Solve challenge 2 of Preppin' Data 2022.

    Parameters
    ----------
    input_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    tuple[pl.DataFrame]
        A tuple containing DataFrames for pupil birthdays in 2022 and the
        number of birthdays per month and weekday.
    """

    # Preprocess the data
    pre_data = challenge01.preprocess_pupil_contact_data(input_fsrc)

    # Normalise the pupil data
    pupil = pre_data.pipe(challenge01.normalize_pupil)

    # Collect the outputs
    pupil_birthday = pupil.pipe(view_pupil_birthday, CALENDAR_YEAR)

    num_birthdays_per_month_weekday = pupil.pipe(
        aggregate_num_birthdays_per_month_weekday, CALENDAR_YEAR
    )

    return (
        pupil_birthday.collect(),
        num_birthdays_per_month_weekday.collect(),
    )


def view_pupil_birthday(pupil: pl.LazyFrame, calendar_year: int) -> pl.LazyFrame:
    """Report pupil birthdays for a specific calendar year.

    Parameters
    ----------
    pupil : pl.LazyFrame
        LazyFrame representing the normalized pupil data.
    calendar_year : int
        The year to consider for birthdays.

    Returns
    -------
    pl.LazyFrame
        DataFrame containing the pupil details, their birthday, and the
        weekday their birthday will fall on.

    Notes
    -----
    Primary key is {pupil_id}
    """

    # Expressions
    born_on = pl.col("born_on")

    this_year_birthday_date_str_expr = (
        str(calendar_year)
        + "-"
        + born_on.dt.month().cast(pl.Utf8)
        + "-"
        + born_on.dt.day().cast(pl.Utf8)
    )

    this_year_birthday_expr = this_year_birthday_date_str_expr.str.to_date()

    birthday_month_expr = born_on.dt.strftime("%B")

    birthday_weekday_expr = this_year_birthday_expr.dt.strftime("%A")

    return pupil.select(
        "pupil_id",
        "first_name",
        "last_name",
        birthday=this_year_birthday_expr,
        birthday_month=birthday_month_expr,
        birthday_weekday=birthday_weekday_expr,
    )


def aggregate_num_birthdays_per_month_weekday(
    pupil: pl.LazyFrame, calendar_year: int
) -> pl.LazyFrame:
    """Aggregate the number of birthdays per month and weekday.

    Parameters
    ----------
    pupil : pl.LazyFrame
        LazyFrame representing the normalized pupil data.
    calendar_year : int
        The year to consider for birthdays.

    Returns
    -------
    pl.LazyFrame
        DataFrame containing the count of birthdays per month and weekday.

    Notes
    -----
    Primary key is {birthday_month, birthday_weekday}
    """

    # Collect the data
    pupil_birthday = pupil.pipe(view_pupil_birthday, calendar_year)

    return pupil_birthday.group_by("birthday_month", "birthday_weekday").agg(
        pl.len().alias("count")
    )
