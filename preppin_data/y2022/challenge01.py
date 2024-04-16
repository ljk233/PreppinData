"""2022: Week 1 - The Prep School - Parental Contact Details

Inputs
======
- __input/2022/PD 2022 Wk 1 Input - Input.csv

Outputs
=======
- output/2022/wk1_pupil_preferred_parent_contact_details.ndjson

ChatGPT's review of the challenge
=================================

The main themes of the challenge appear to revolve around data preprocessing,
data reshaping, and data analysis.

1. Data Preprocessing:
   The challenge involves loading data from a CSV file, cleaning the data
   by renaming columns and converting data types (e.g., converting date
   strings to date objects), and handling missing or inconsistent data.

2. Data Reshaping:
   The data is reshaped [...] to transform wide-format data into long-format
   data. This reshaping is essential for further analysis, as it allows
   for easier manipulation and aggregation of the data.

3. Data Normalization:
   The solution includes normalizing the data by removing redundant columns
   and ensuring data consistency. This step helps in reducing redundancy
   and preparing the data for analysis.

4. Data Analysis and Reporting:
   The final part of the challenge involves analyzing the normalized data
   to report pupil preferred parent contact details. This includes joining
   multiple data frames, filtering data based on certain criteria, and
   selecting specific columns for the final output.

5. Primary Keys and Data Relationships:
   Throughout the solution, there is a focus on understanding the primary
   keys and relationships between different data frames. This understanding
   is crucial for performing joins and ensuring the accuracy of the analysis.

Overall, the challenge covers various aspects of data manipulation and analysis,
from initial preprocessing to final reporting, making it a comprehensive
exercise in data preparation and analysis.
"""

from datetime import date

import polars as pl


def solve(input_fsrc: str) -> pl.DataFrame:
    """Solves challenge 1 of Preppin' Data 2022.

    Parameters
    ----------
    input_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.DataFrame
        A DataFrame showing each pupil's preferred contact details.

    Notes
    -----
    """
    # Preprocess the data
    pre_data = preprocess_data(input_fsrc)

    # Normalise the data
    pupil = pre_data.pipe(normalize_pupil)

    pupil_contact = pre_data.pipe(normalize_pupil_contact)

    return report_pupil_preferred_parent_contact_details(pupil, pupil_contact).collect()


def preprocess_data(input_fsrc: str) -> pl.LazyFrame:
    """Preprocesses the input data.

    Parameters
    ----------
    input_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.LazyFrame
        Preprocessed data.

    Notes
    -----
    Primary key is {pupil_id, parental_contact_number}
    """
    return load_data(input_fsrc).pipe(reshape_data).pipe(clean_data)


def load_data(input_fsrc: str) -> pl.LazyFrame:
    """Load the input data from the given path.

    Parameters
    ----------
    input_fsrc : str
        Filepath of the input CSV file.

    Returns
    -------
    pl.LazyFrame
        Loaded data.
    """
    return pl.scan_csv(input_fsrc)


def reshape_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Reshape the input data.

    Parameters
    ----------
    data : pl.LazyFrame
        Input data.

    Returns
    -------
    pl.LazyFrame
        Reshaped data.
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


def clean_data(reshaped_data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean the reshaped data.

    Parameters
    ----------
    reshaped_data : pl.LazyFrame
        Reshaped data.

    Returns
    -------
    pl.LazyFrame
        Cleaned data.
    """
    col_mapper = {
        "id": "pupil_id",
        "pupil first name": "first_name",
        "pupil last name": "last_name",
        "Date of Birth": "born_on",
        "Preferred Contact Employer": "preferred_contact_employer",
        "Parental Contact": "preferred_contact_number",
    }

    return reshaped_data.with_columns(
        pl.col("Date of Birth").str.to_date("%m/%d/%Y")
    ).rename(col_mapper)


def normalize_pupil(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Normalize the pupil data.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        Preprocessed data.

    Returns
    -------
    pl.LazyFrame
        Normalized pupil data.

    Notes
    -----
    Primary key is {pupil_id}
    """
    return pre_data.drop(
        "parental_contact_number", "parental_contact_first_name"
    ).unique("pupil_id")


def normalize_pupil_contact(pre_data: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize the pupil contact data.

    Parameters
    ----------
    pre_data : pl.LazyFrame
        Preprocessed data.

    Returns
    -------
    pl.LazyFrame
        Normalized pupil contact data.

    Notes
    -----
    Primary key is {pupil_id, parental_contact_number}
    """
    return pre_data.select(
        "pupil_id",
        pl.col("parental_contact_number").str.extract(r"(\d+)$").cast(pl.Int64),
        "parental_contact_first_name",
    )


def report_pupil_preferred_parent_contact_details(
    pupil: pl.LazyFrame,
    pupil_contact: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Reports pupil preferred parent contact details.

    Parameters
    ----------
    pupil : pl.LazyFrame
        Pupil data.
    pupil_contact : pl.LazyFrame
        Pupil contact data.

    Returns
    -------
    pl.LazyFrame
        Pupil preferred parent contact details.
    """
    pupil_academic_year = pupil.pipe(view_pupil_academic_year)

    pupil_full_name = pupil.pipe(view_pupil_full_name)

    selected_pupil_contact = view_pupil_parental_contact_details(pupil, pupil_contact)

    return (
        pupil.join(pupil_academic_year, on="pupil_id")
        .join(pupil_full_name, on="pupil_id")
        .join(selected_pupil_contact, on="pupil_id")
        .filter(pl.col("preferred_contact_number") == pl.col("parental_contact_number"))
        .select(
            "academic_year",
            "pupil_full_name",
            "parental_contact_full_name",
            "parental_contact_email",
        )
    )


def view_pupil_academic_year(pupil: pl.LazyFrame) -> pl.LazyFrame:
    """Derive the pupil's academic year based on their 'born_on' date.

    Parameters
    ----------
    pupil : pl.LazyFrame
        Pupil data.

    Returns
    -------
    pl.LazyFrame
        Pupil academic year.
    """
    born_on = pl.col("born_on")

    academic_year_expr = (
        pl.when(born_on >= date(2014, 9, 1))
        .then(pl.lit(1))
        .when(born_on >= date(2013, 9, 1))
        .then(pl.lit(2))
        .when(born_on >= date(2012, 9, 1))
        .then(pl.lit(3))
        .when(born_on >= date(2011, 9, 1))
        .then(pl.lit(4))
    )

    return pupil.select("pupil_id", academic_year_expr.alias("academic_year"))


def view_pupil_full_name(pupil: pl.LazyFrame) -> pl.LazyFrame:
    """Derive the pupil's full name.

    Parameters
    ----------
    pupil : pl.LazyFrame
        Pupil data.

    Returns
    -------
    pl.LazyFrame
        Pupil full name.
    """
    return pupil.select(
        "pupil_id",
        (pl.col("first_name") + " " + pl.col("last_name")).alias("pupil_full_name"),
    )


def view_pupil_parental_contact_details(
    pupil: pl.LazyFrame,
    pupil_contact: pl.LazyFrame,
) -> pl.LazyFrame:
    """View the pupil's parental contact details.

    Parameters
    ----------
    pupil : pl.LazyFrame
        Pupil data.
    pupil_contact : pl.LazyFrame
        Pupil contact data.

    Returns
    -------
    pl.LazyFrame
        Pupil parental contact details.
    """
    return pupil.join(
        pupil_contact,
        on="pupil_id",
    ).select(
        "pupil_id",
        "parental_contact_number",
        (pl.col("parental_contact_first_name") + " " + pl.col("last_name")).alias(
            "parental_contact_full_name"
        ),
        (
            pl.col("parental_contact_first_name")
            + "."
            + pl.col("last_name")
            + "@"
            + pl.col("preferred_contact_employer")
            + ".com"
        )
        .str.to_lowercase()
        .alias("parental_contact_email"),
    )
