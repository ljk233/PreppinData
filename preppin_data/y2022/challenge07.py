"""2022: Week 7 - Call Centre Agent Metrics

Inputs
------
- __input/2022/MetricData2021.xlsx
- __input/2022/PeopleData.xlsx

Outputs
-------
- output/2022/wk07_agent_performance.ndjson
"""

import polars as pl


CALENDAR_YEAR = 2021


def solve(metric_data_fsrc: str, peope_data_fsrc: str) -> pl.DataFrame:
    """Solve challenge 7 of Preppin' Data 2022.

    Parameters
    ----------
    metric_data_fsrc : str
        Filepath of the input XLSX file containing metric data.
    peope_data_fsrc : str
        Filepath of the input XLSX file containing people data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the analysis of agent monthly performance.

    Notes
    -----
    This function:
    - Loads the necessary data from input Excel files
    - Preprocesses the data contained in the input Excel files.
    - Returns a DataFrame with the required analysis results.
    """

    # Load the data
    metric_data_dict = load_data(metric_data_fsrc)

    peope_data_dict = load_data(peope_data_fsrc)

    # Preprocess the data
    agent_metric = preprocess_agent_metric_data(
        metric_data_dict,
        # METRIC_NAME,
        CALENDAR_YEAR,
    )

    agent = peope_data_dict["People"].pipe(preprocess_agent_data)

    leader = peope_data_dict["Leaders"].pipe(preprocess_leader_data)

    location = peope_data_dict["Location"].pipe(preprocess_location_data)

    # Collect the data
    agent_monthly_performance = view_agent_performance(
        agent_metric,
        agent,
        leader,
        location,
    )

    return agent_monthly_performance


def load_data(fsrc: str) -> dict[str, pl.DataFrame]:
    """Load data from the input Excel file.

    Parameters
    ----------
    fsrc : str
        Filepath of the input XLSX file.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary containing Polars DataFrames with loaded data. The
        keys are the names of the sheets in the Excel file, and the values
        are the corresponding DataFrames.
    """

    data_dict = pl.read_excel(fsrc, sheet_id=0)

    return data_dict


def preprocess_agent_metric_data(
    metric_data_dict: dict[str, pl.DataFrame], year: int
) -> pl.DataFrame:
    """Preprocess the monthly agent metric data dictionary.

    Parameters
    ----------
    metric_data_dict : dict[str, pl.DataFrame]
        A dictionary containing Polars DataFrames with loaded agent monthly
        metric data.
    year : int
        Calendar year the metric data was collected.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed agent metric data.
    """

    pre_data_arr = []

    for sheet_name, data in metric_data_dict.items():
        harmonized_data = data.pipe(harmonize_monthly_agent_metric_data)

        pre_data = harmonized_data.pipe(clean_monthly_metric_data, sheet_name, year)

        pre_data_arr.append(pre_data)

    return pl.concat(pre_data_arr, how="diagonal")


def harmonize_monthly_agent_metric_data(data: pl.DataFrame) -> pl.DataFrame:
    """Harmonize the monthly agent metric data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the monthly agent metric data for some month
        from the input XLSX file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the harmonized monthly agent metric data.
    """

    # Expressions
    orig_metric_name = pl.col("orig_metric_name")

    new_metric_name_expr = (
        pl.when(orig_metric_name.str.contains("Offered"))
        .then(pl.lit("num_calls_offered"))
        .when(orig_metric_name.str.contains("Not Answered"))
        .then(pl.lit("num_calls_not_answered"))
        .when(orig_metric_name.str.contains("Answered"))
        .then(pl.lit("num_calls_answered"))
        .when(orig_metric_name == "Total Duration")
        .then(pl.lit("total_duration"))
        .when(orig_metric_name == "Sentiment")
        .then(pl.lit("call_sentiment"))
        .when(orig_metric_name == "Transfers")
        .then(pl.lit("num_calls_transferred"))
        .otherwise(pl.lit(None))
    )

    return (
        data.melt(id_vars="AgentID", variable_name="orig_metric_name")
        .with_columns(new_metric_name=new_metric_name_expr)
        .pivot(
            values="value",
            index="AgentID",
            columns="new_metric_name",
        )
    )


def clean_monthly_metric_data(
    harmonized_data: pl.DataFrame,
    month_name: str,
    year: int,
) -> pl.DataFrame:
    """Clean the harmonized monthly agent metric data.

    Parameters
    ----------
    pl.DataFrame
        DataFrame containing the harmonized monthly agent metric data.
    month_name : str
        Month the metric data was collected as locale's abbreviated name.
    year : int
        Calendar year the metric data was collected.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the cleansed monthly agent metric data.
    """

    col_mapper = {"AgentID": "agent_id"}

    # Expressions
    month_name_expr = pl.lit(month_name)

    year_expr = pl.lit(year).cast(pl.Utf8)

    collected_in_expr = month_name_expr + "-" + year_expr

    month_number_expr = ("01-" + collected_in_expr).str.to_date("%d-%b-%Y").dt.month()

    return harmonized_data.with_columns(
        collected_in=collected_in_expr, month_number=month_number_expr
    ).rename(col_mapper)


def preprocess_agent_data(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the agent data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the agent data from the input XLSX file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed agent data.
    """

    return data.pipe(reshape_agent_data).pipe(clean_agent_data)


def reshape_agent_data(data: pl.DataFrame) -> pl.DataFrame:
    """Reshape the input agent data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the agent data from the input XLSX file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the reshaped agent data.
    """

    # Expressions
    leader_number_expr = pl.col("variable").str.extract(r"(\d+)").cast(pl.Int64)

    return (
        data.melt(
            id_vars=["id", "first_name", "last_name", "Location ID"],
            value_name="leader_id",
        )
        .with_columns(leader_number=leader_number_expr)
        .drop("variable")
    )


def clean_agent_data(reshaped_data: pl.DataFrame) -> pl.DataFrame:
    """Clean the reshaped agent data.

    Parameters
    ----------
    reshaped_data : pl.DataFrame
        DataFrame containing the reshaped agent data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the cleansed agent data.
    """

    col_mapper = {
        "id": "agent_id",
        "first_name": "agent_first_name",
        "last_name": "agent_last_name",
        "Location ID": "location_id",
    }

    return reshaped_data.rename(col_mapper)


def preprocess_leader_data(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the leader data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the leader data from the input XLSX file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed leader data.
    """

    col_mapper = {
        "id": "leader_id",
        "first_name": "leader_first_name",
        "last_name": "leader_last_name",
    }

    return data.rename(col_mapper)


def preprocess_location_data(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the location data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the location data from the input XLSX file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed location data.
    """

    col_mapper = {"Location ID": "location_id", "Location": "location_name"}

    return data.rename(col_mapper)


def view_agent_performance(
    agent_metric: pl.DataFrame,
    agent: pl.DataFrame,
    leader: pl.DataFrame,
    location: pl.DataFrame,
) -> pl.DataFrame:
    """View the agent monthly performance summaries.

    Parameters
    ----------
    agent_metric : pl.DataFrame
        DataFrame containing the monthly metric data for agents.
    agent : pl.DataFrame
        DataFrame containing the preprocessed agent data.
    leader : pl.DataFrame
        DataFrame containing the preprocessed leader data.
    location : pl.DataFrame
        DataFrame containing the preprocessed location data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the agent monthly performance summaries.

    Notes
    -----
    This function generates a summary of the monthly performance for each
    agent, including metrics such as call response rate, average call duration,
    and sentiment analysis results. It combines data from multiple DataFrames
    to create a comprehensive overview of agent performance.
    """

    # Collect the data
    agent_monthly_performance = agent_metric.pipe(scaffold_agent_monthly_performance)

    # Expressions
    rate_calls_not_answered_expr = pl.col("num_calls_not_answered") / pl.col(
        "num_calls_offered"
    )

    mean_call_duration_expr = pl.col("total_duration") / pl.col("num_calls_answered")

    had_postive_call_sentiment_expr = (
        pl.when(pl.col("call_sentiment") >= 0)
        .then(pl.lit(True))
        .when(pl.col("call_sentiment") < 0)
        .then(pl.lit(False))
        .otherwise(pl.lit(None))
    )

    did_meet_rate_call_not_answered_threshold_expr = (
        pl.when(rate_calls_not_answered_expr >= 0.05)
        .then(pl.lit(False))
        .when(rate_calls_not_answered_expr < 0.05)
        .then(pl.lit(True))
        .otherwise(pl.lit(None))
    )

    return (
        agent_monthly_performance.join(
            agent_metric,
            on=[
                "agent_id",
                "month_number",
                "collected_in",
            ],
            how="left",
        )
        .join(agent, on="agent_id")
        .join(leader, on="leader_id")
        .join(location, on="location_id")
        .filter(pl.col("leader_number") == 1)
        .select(
            "collected_in",
            "month_number",
            "agent_id",
            agent_full_name=full_name_expr("agent_first_name", "agent_last_name"),
            leader_full_name=full_name_expr("leader_first_name", "leader_last_name"),
            rate_calls_not_answered=rate_calls_not_answered_expr.round(3),
            mean_call_duration=mean_call_duration_expr.round(1),
            had_postive_call_sentiment=had_postive_call_sentiment_expr,
            did_meet_call_not_answered_rate_threshold=(
                did_meet_rate_call_not_answered_threshold_expr
            ),
        )
        .sort("agent_id", "month_number")
    )


def scaffold_agent_monthly_performance(agent_metric: pl.DataFrame) -> pl.DataFrame:
    """Scaffold the agent monthly performance data.

    Parameters
    ----------
    agent_metric : pl.DataFrame
        DataFrame containing the agent metric data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the scaffolded agent monthly performance data,
        with one row for each agent for each month.

    Notes
    -----
    This function generates a DataFrame that serves as the basis for agent
    monthly performance analysis. It creates a Cartesian product of unique
    agent IDs and unique month identifiers.
    """

    agent = agent_metric.select("agent_id").unique()

    month = agent_metric.select("month_number", "collected_in").unique()

    return agent.join(month, how="cross")


def full_name_expr(first_name: str, last_name: str) -> pl.Expr:
    """Create a Polars expression to concatenate first name and last name
    into a full name.

    Parameters
    ----------
    first_name : str
        The column name or string representing the first name.
    last_name : str
        The column name or string representing the last name.

    Returns
    -------
    pl.Expr
        A Polars expression that represents the full name concatenation.
    """

    return pl.col(last_name) + ", " + pl.col(first_name)
