"""2022: Week 1 - The Prep School - Parental Contact Details

Inputs
------
- __input/2022/PD 2022 Wk 1 Input - Input.csv

Outputs
-------
- output/2022/wk01_pupil_preferred_parent_contact_details.ndjson
"""

from datetime import date

import polars as pl


def solve(pd_2022_wk1_fsrc: str) -> pl.DataFrame:
    """Solve challenge 1 of Preppin' Data 2022.

    Parameters
    ----------
    pd_2022_wk1_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing each pupil's preferred contact details.
    """

    # Preprocess the data
    pre_data = preprocess_pupil_contact_data(pd_2022_wk1_fsrc)

    # Normalise the data
    pupil = pre_data.pipe(normalize_pupil)

    parent_contact = pre_data.pipe(normalize_parental_contact)

    return view_preferred_parental_contact_details(pupil, parent_contact).collect()


def preprocess_pupil_contact_data(pd_2022_wk1_fsrc: str) -> pl.LazyFrame:
    """Preprocesses the pupil contact data.

    Parameters
    ----------
    pd_2022_wk1_fsrc : str
        Filepath of the pupil contact CSV file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed pupil contact data.

    Notes
    -----
    Primary key is {pupil_id, parental_contact_number}
    """

    data = load_data(pd_2022_wk1_fsrc)

    return data.pipe(reshape_pupil_contact_data).pipe(clean_pupil_contact_data)


def load_data(pd_2022_wk1_fsrc: str) -> pl.LazyFrame:
    """Load the input data from the given path.

    Parameters
    ----------
    pd_2022_wk1_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the input data.
    """

    return pl.scan_csv(pd_2022_wk1_fsrc)


def reshape_pupil_contact_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Reshape the pupil contact data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the pupil contact data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the reshaped pupil contact data.
    """

    return data.melt(
        id_vars=[
            "id",
            "pupil first name",
            "pupil last name",
            "gender",
            "Date of Birth",
            "Preferred Contact Employer",
            "Parental Contact",
        ],
        variable_name="parental_contact_number",
        value_name="parental_contact_first_name",
    )


def clean_pupil_contact_data(reshaped_data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the reshaped pupil contact data.

    Parameters
    ----------
    reshaped_data : pl.LazyFrame
        LazyFrame representing the reshaped pupil contact data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the cleaned pupil contact data.
    """

    col_mapper = {
        "id": "pupil_id",
        "pupil first name": "first_name",
        "pupil last name": "last_name",
        "Date of Birth": "born_on",
        "Preferred Contact Employer": "preferred_parental_contact_employer",
        "Parental Contact": "preferred_parental_contact_number",
    }

    # Expressions
    date_of_birth_expr = pl.col("Date of Birth").str.to_date("%m/%d/%Y")

    return reshaped_data.with_columns(date_of_birth_expr).rename(col_mapper)


def normalize_pupil(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize the pupil data.

    Parameters
    ----------
    pl.LazyFrame
        LazyFrame representing the preprocessed pupil contact data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the normalized pupil data.

    Notes
    -----
    Primary key is {pupil_id}
    """

    return pre_data.drop(
        "parental_contact_number", "parental_contact_first_name"
    ).unique("pupil_id")


def normalize_parental_contact(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize the pupil contact data.

    Parameters
    ----------
    pl.LazyFrame
        LazyFrame representing the preprocessed input data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the normalized pupil contact data.

    Notes
    -----
    Primary key is {pupil_id, parental_contact_number}
    """

    # Expressions
    parental_contact_number_expr = (
        pl.col("parental_contact_number").str.extract(r"(\d+)$").cast(pl.Int64)
    )

    return pre_data.select(
        "pupil_id",
        parental_contact_number_expr,
        "parental_contact_first_name",
    )


def view_preferred_parental_contact_details(
    pupil: pl.LazyFrame, parental_contact: pl.LazyFrame
) -> pl.LazyFrame:
    """View pupil's preferred parental contact details.

    Parameters
    ----------
    pupil : pl.LazyFrame
        LazyFrame representing the normalized pupil data.
    parental_contact : pl.LazyFrame
        LazyFrame representing the normalized parental contact data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing pupil's preferred parental contact details.
    """

    # Expressions
    academic_year_expr = (
        pl.when(pl.col("born_on") >= date(2014, 9, 1))
        .then(pl.lit(1))
        .when(pl.col("born_on") >= date(2013, 9, 1))
        .then(pl.lit(2))
        .when(pl.col("born_on") >= date(2012, 9, 1))
        .then(pl.lit(3))
        .when(pl.col("born_on") >= date(2011, 9, 1))
        .then(pl.lit(4))
    )

    pupil_full_name_expr = pl.col("first_name") + " " + pl.col("last_name")

    parental_contact_full_name_expr = (
        pl.col("parental_contact_first_name") + " " + pl.col("last_name")
    )

    parental_contact_email_expr = (
        pl.col("parental_contact_first_name")
        + "."
        + pl.col("last_name")
        + "@"
        + pl.col("preferred_parental_contact_employer")
        + ".com"
    ).str.to_lowercase()

    return pupil.join(
        parental_contact,
        left_on=["pupil_id", "preferred_parental_contact_number"],
        right_on=["pupil_id", "parental_contact_number"],
    ).select(
        academic_year=academic_year_expr,
        pupil_full_name=pupil_full_name_expr,
        parental_contact_full_name=parental_contact_full_name_expr,
        parental_contact_email=parental_contact_email_expr,
    )
