"""2023: Week 8 - Taking Stock

Inputs
------
- __input/2023

Outputs
-------
- output/2023/wk08_monthly_top_five_trades.ndjson
"""

from glob import glob

import polars as pl


def solve(input_dir_path: str) -> pl.DataFrame:
    """Solve challenge 8 of Preppin' Data 2023.

    Parameters
    ----------
    input_dir_path : str
        Directory path containing input files.

    Returns
    -------
    pl.DataFrame
        Preprocessed and analyzed trade data.

    Notes
    -----
    This function follows these steps:
    1. Load trade data from CSV files matching the pattern 'MOCK_DATA' in
       the input directory.
    2. Preprocess the trade data by cleaning, annotating, and categorizing
       it.
    3. View and extract the top five monthly trades based on market capitalization
       and purchase price.
    """

    # Load the data
    data_dict = load_trade_data(input_dir_path, patt="MOCK_DATA")

    # Preprocess the data
    pre_trade = preprocess_trade_data(data_dict)

    return pre_trade.pipe(view_top_five_monthly_trades).collect()


def load_trade_data(input_dir_path: str, patt: str) -> dict[str, pl.LazyFrame]:
    """Load trade data from CSV files matching a pattern in the given directory.

    Parameters
    ----------
    input_dir_path : str
        Directory path containing input files.
    patt : str
        Pattern to match filenames.

    Returns
    -------
    dict[str, pl.LazyFrame]
        A dictionary with file paths as keys and LazyFrames as values.
    """

    file_paths = collect_file_paths(input_dir_path, patt)

    return {
        file_path: pl.scan_csv(file_path, null_values=["n/a"])
        for file_path in file_paths
    }


def collect_file_paths(dir_path: str, patt: str) -> list[str]:
    """Collect file paths matching a pattern in a directory.

    Parameters
    ----------
    dir_path : str
        Directory path to search for files.
    patt : str
        Pattern to match filenames.

    Returns
    -------
    list[str]
        List of file paths.
    """

    file_ls = glob(rf"{patt}*.csv", root_dir=dir_path)

    return [f"{dir_path}/{f}" for f in file_ls]


def preprocess_trade_data(data_dict: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Preprocess the trade data.

    Parameters
    ----------
    data_dict : dict[str, pl.LazyFrame]
        Dictionary containing file paths and LazyFrames.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the preprocessed trade data.

    Notes
    -----
    Primary key is {trade_id}
    """

    # Collect the data
    data_arr = [
        data.pipe(clean_trade_data, file_path) for file_path, data in data_dict.items()
    ]

    return pl.concat(data_arr).pipe(reset_primary_key)


def clean_trade_data(data: pl.LazyFrame, file_path: str) -> pl.LazyFrame:
    """Clean the trade data.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the trade data.
    file_path : str
        Path of the file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the cleansed trade data.
    """

    col_mapper = {
        "Ticker": "ticker",
        "Sector": "sector",
        "Market": "market",
        "Stock Name": "stock_name",
        "Market Cap": "market_cap",
        "Purchase Price": "purchase_price",
    }

    return (
        data.pipe(annotate_file_creation, file_path)
        .pipe(clean_market_cap)
        .pipe(clean_purchase_price)
        .drop_nulls(["Purchase Price", "Market Cap"])
        .drop("file_path")
        .rename(col_mapper)
    )


def annotate_file_creation(data: pl.LazyFrame, file_path: str) -> pl.LazyFrame:
    """Annotate trade data with file creation date.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the trade data.
    file_path : str
        Path of the file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the trade data with annotated file creation
        date.
    """

    # Expressions
    file_path_expr = pl.lit(file_path)

    file_month_expr = file_path_expr.str.extract(r"MOCK_DATA-(\d+)").fill_null("1")

    file_date_expr = ("2023-" + file_month_expr + "-1").str.to_date()

    return data.with_columns(file_date=file_date_expr)


def clean_market_cap(data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean Market Cap column.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the trade data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the trade data with cleansed Market Cap
        column.
    """

    # Expressions
    market_cap_num_expr = (
        pl.col("Market Cap").str.extract(r"(\d+.\d+|\d+)").cast(pl.Float64)
    )

    market_cap_multiplier_expr = (
        pl.when(pl.col("Market Cap").str.ends_with("K"))
        .then(pl.lit(1000))
        .when(pl.col("Market Cap").str.ends_with("M"))
        .then(pl.lit(1_000_000))
        .when(pl.col("Market Cap").str.ends_with("B"))
        .then(pl.lit(1_000_000_000))
        .otherwise(pl.lit(1))
    )

    market_cap_expr = (market_cap_num_expr * market_cap_multiplier_expr).cast(pl.Int64)

    return data.with_columns(market_cap_expr)


def clean_purchase_price(data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean Purchase Price column.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the trade data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the trade data with cleansed Purchase Price
        column.
    """

    # Expressions
    purchase_price_expr = (
        pl.col("Purchase Price").str.strip_prefix("$").cast(pl.Float64)
    )

    return data.with_columns(purchase_price_expr)


def reset_primary_key(data: pl.LazyFrame) -> pl.LazyFrame:
    """Reset primary key to be globally unique.

    Parameters
    ----------
    data : pl.LazyFrame
        LazyFrame representing the trade data.

    Returns
    -------
    pl.LazyFrame
        Trade data with reset primary key.
    """

    return data.drop("id").with_row_index("trade_id", offset=1)


def view_top_five_monthly_trades(pre_trade: pl.LazyFrame) -> pl.LazyFrame:
    """View top five monthly trades.

    Parameters
    ----------
    pre_trade : pl.LazyFrame
        LazyFrame representing the preprocessed trade data.

    Returns
    -------
    pl.LazyFrame
        Top five monthly trades.
    """

    return (
        pre_trade.pipe(categorize_trade)
        .pipe(rank_categorized_trade)
        .filter(pl.col("rank") <= 5)
        .drop(
            "first_name",
            "last_name",
            "trade_id",
        )
        .sort(
            "file_date",
            "market_cap_category",
            "purchase_price_category",
            "rank",
        )
    )


def categorize_trade(pre_trade: pl.LazyFrame) -> pl.LazyFrame:
    """Categorize trade data based on Market Cap and Purchase Price.

    Parameters
    ----------
    pre_trade : pl.LazyFrame
        LazyFrame representing the preprocessed trade data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the categorized trade data.
    """

    # Expressions
    market_cap_category_expr = (
        pl.when(pl.col("market_cap") < 100_000_000)
        .then(pl.lit("Small"))
        .when(pl.col("market_cap") < 1_000_000_000)
        .then(pl.lit("Medium"))
        .when(pl.col("market_cap") < 100_000_000_000)
        .then(pl.lit("Large"))
        .otherwise(pl.lit("Huge"))
    )

    purchase_price_category_expr = (
        pl.when(pl.col("purchase_price") < 25_000)
        .then(pl.lit("Small"))
        .when(pl.col("purchase_price") < 50_000)
        .then(pl.lit("Medium"))
        .when(pl.col("purchase_price") < 75_000)
        .then(pl.lit("Large"))
        .otherwise(pl.lit("Very Large"))
    )

    return pre_trade.with_columns(
        market_cap_category=market_cap_category_expr,
        purchase_price_category=purchase_price_category_expr,
    )


def rank_categorized_trade(categorized_data: pl.LazyFrame) -> pl.LazyFrame:
    """Rank the purchase price by the file date, market cap category, and
    purchase price category.

    Parameters
    ----------
    categorized_data : pl.LazyFrame
        LazyFrame representing the categorized trade data.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the ranked trade data.
    """

    rank_expr = (
        pl.col("purchase_price")
        .rank("min")
        .over(
            "file_date",
            "market_cap_category",
            "purchase_price_category",
        )
    )

    return categorized_data.with_columns(rank=rank_expr).sort(
        "file_date",
        "market_cap_category",
        "purchase_price_category",
        "rank",
    )
