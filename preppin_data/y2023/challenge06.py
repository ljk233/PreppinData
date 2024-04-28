"""2023: Week 6 - DSB Customer Ratings

Inputs
------
- __input/2023/DSB Customer Survery.csv

Outputs
-------
- output/2023/wk06_dsb_customer_ratings.ndjson

Notes
-----
The rating metric in the customer survey responses is based on a Likert
scale, which represents ordinal values rather than continuous measurements.
Therefore, when aggregating the customer responses by platform in the
`aggregate_customer_response` function, the median value is used instead
of the mean. This approach is more appropriate for Likert scale data as
it helps to mitigate the influence f outliers and skewness, providing a
more robust summary of customer perceptions. Additionally, since the rating
is an ordinal value rather than a true numerical value, the median offers
a better representation of the central tendency of the data.
"""

import polars as pl


def solve(customer_survey_fsrc: str) -> pl.DataFrame:
    """Solve challenge 5 of Preppin' Data 2023.

    Parameters
    ----------
    customer_survey_fsrc : str
        Filepath of the input CSV file containing customer survey responses
        data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the analysis of the customer survey responses.

    Notes
    -----
    This function loads the customer survey responses from a CSV file,
    preprocesses it, performs analysis on the preprocessed data, and returns
    the results.
    """

    # Load the data
    survey_response = load_survey_responses_data(customer_survey_fsrc)

    # Preprocess the data
    pre_survey_response = survey_response.pipe(preprocess_survey_response_data)

    return pre_survey_response.pipe(view_survey_response_analysis)


def load_survey_responses_data(customer_survey_fsrc: str) -> pl.DataFrame:
    """Load the customer survey responses from a CSV file.

    Parameters
    ----------
    customer_survey_fsrc : str
        Filepath of the input CSV file containing customer survey responses
        data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the loaded customer survey responses.

    Notes
    -----
    The function returns a DataFrame instead of a LazyFrame. This decision
    was made because the analysis requires pivoting the data, which would
    necessitate collecting the LazyFrame to a DataFrame midway through the
    process, which would mean an unexpected change in return types. Maintaining
    consistency in return types throughout the module helps ensure predictable
    behavior and facilitates code comprehension.
    """

    return pl.read_csv(customer_survey_fsrc)


def preprocess_survey_response_data(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the customer survey responses.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the loaded customer survey responses.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed customer survey responses.

    Notes
    -----
    This function performs the following actions:
    - Reshapes the data.
    - Cleans the data.

    Primary key is {person_id, platform, category}
    """

    return data.pipe(reshape_survey_response_data).pipe(clean_survey_response_data)


def reshape_survey_response_data(data: pl.DataFrame) -> pl.DataFrame:
    """Reshape the customer survey responses.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the loaded customer survey responses.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the reshaped customer survey responses.
    """

    return data.melt(id_vars="Customer ID")


def clean_survey_response_data(reshaped_data: pl.DataFrame) -> pl.DataFrame:
    """Clean the reshaped customer survey data.

    Parameters
    ----------
    reshaped_data : pl.DataFrame
        DataFrame containing the reshaped customer survey data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the cleansed new customer data.

    Notes
    -----
    This function performs data cleaning on the reshaped customer survey data,
    including:
    1. Renaming columns.
    2. Extracting information from text.
    3. Filtering out the incorect "Overall Rating" data.
    4. Dropping unnecessary columns.
    """

    col_mapper = {"Customer ID": "customer_id", "value": "rating"}

    # Expressions
    platform_expr = pl.col("variable").str.extract(r"^(.+) -")

    category_expr = pl.col("variable").str.extract(r"- (.+)$")

    return (
        reshaped_data.with_columns(
            platform=platform_expr,
            category=category_expr,
        )
        .filter(pl.col("category") != "Overall Rating")
        .rename(col_mapper)
        .drop("variable")
    )


def view_survey_response_analysis(pre_survey_response: pl.DataFrame) -> pl.DataFrame:
    """View the customer survey responses analysis.

    Parameters
    ----------
    pre_survey_response : pl.DataFrame
        DataFrame containing the preprocessed customer survey responses.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the analysis of the customer survey responses.

    Notes
    -----
    This function computes and returns an analysis of the customer survey
    responses. It counts the number of responses for each category and
    calculates the percentage of responses in each category relative to
    the total number of responses.
    """

    # Collect the categorised customer response
    categorised_customer_response = pre_survey_response.pipe(
        view_categorised_survey_response
    )

    # Expressions
    num_responses_expr = pl.len()

    total_responses_expr = pl.sum("num_responses")

    pct_responses_expr = 100 * pl.col("num_responses") / total_responses_expr

    return (
        categorised_customer_response.group_by("category")
        .agg(num_responses=num_responses_expr)
        .with_columns(pct_responses=pct_responses_expr)
        .sort("pct_responses", descending=True)
    )


def view_categorised_survey_response(
    pre_survey_response: pl.DataFrame,
) -> pl.DataFrame:
    """View the categorised customer survey responses.

    Parameters
    ----------
    pre_survey_response : pl.DataFrame
        DataFrame containing the preprocessed customer survey responses
        data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the categorised customer survey responses.

    Notes
    -----
    This function categorizes the customer survey responses based on the
    difference between ratings for the "Mobile App" and "Online Interface"
    platforms. It classifies customers as "Mobile App Superfan", "Mobile App Fan",
    "Online Interface Superfan", "Online Interface Fan", or "Neutral" based
    on predefined criteria.
    """

    # Expressions
    difference_expr = pl.col("Mobile App") - pl.col("Online Interface")

    category_expr = (
        pl.when(pl.col("difference").is_between(-1, 1))
        .then(pl.lit("Neutral"))
        .when(pl.col("difference") >= 2)
        .then(pl.lit("Mobile App Superfan"))
        .when(pl.col("difference").is_between(1, 2))
        .then(pl.lit("Mobile App Fan"))
        .when(pl.col("difference") <= -2)
        .then(pl.lit("Online Interface Superfan"))
        .when(pl.col("difference").is_between(-2, -1))
        .then(pl.lit("Online Interface Fan"))
        .otherwise(pl.lit(None))
    )

    return (
        pre_survey_response.pipe(aggregate_survey_response)
        .with_columns(difference=difference_expr)
        .with_columns(category=category_expr)
    )


def aggregate_survey_response(pre_survey_response: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the customer response responses by platform.

    Parameters
    ----------
    pre_customer_survey : pl.DataFrame
        DataFrame containing the preprocessed customer survey responses
        data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the aggregated customer responses.

    Notes
    -----
    This function aggregates the customer responses by platform, computing
    the median rating for each platform based on the customer survey responses
    data. It pivots the data to show the median rating for each customer
    and platform combination.

    The rating metric used is a Likert scale, and so the median value is
    calculated for each platform to summarize the customers' perceptions.
    """

    return pre_survey_response.pivot(
        values="rating",
        index="customer_id",
        columns="platform",
        aggregate_function="median",
    )
