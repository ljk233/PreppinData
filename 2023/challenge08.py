"""2023.challenge08.py
"""

import polars as pl


INPUT_FILE_MONTH_DICT = [
    ("2023/data/input/MOCK_DATA.csv", "Jan-2023"),
    ("2023/data/input/MOCK_DATA-2.csv", "Feb-2023"),
    ("2023/data/input/MOCK_DATA-3.csv", "Mar-2023"),
    ("2023/data/input/MOCK_DATA-4.csv", "Apr-2023"),
    ("2023/data/input/MOCK_DATA-5.csv", "May-2023"),
    ("2023/data/input/MOCK_DATA-6.csv", "Jun-2023"),
    ("2023/data/input/MOCK_DATA-7.csv", "Jul-2023"),
    ("2023/data/input/MOCK_DATA-8.csv", "Aug-2023"),
    ("2023/data/input/MOCK_DATA-9.csv", "Sep-2023"),
    ("2023/data/input/MOCK_DATA-10.csv", "Oct-2023"),
    ("2023/data/input/MOCK_DATA-11.csv", "Nov-2023"),
    ("2023/data/input/MOCK_DATA-12.csv", "Dec-2023"),
]
OUTPUT_NDJSON = "2023/data/output/wk8_ranked_purchases.ndjson"


def main():
    return (
        stack_data(INPUT_FILE_MONTH_DICT)
        .pipe(preprocess_data)
        .pipe(categorize_data)
        .pipe(rank_data)
        .pipe(postprocess_data)
        .collect()
        .write_ndjson(OUTPUT_NDJSON)
    )


def stack_data(input_file_month_months: list[tuple[str, str]]) -> pl.LazyFrame:
    """"""
    return pl.concat(
        load_data(fsrc, month_name) for fsrc, month_name in input_file_month_months
    )


def load_data(fsrc: str, month_name: str) -> pl.LazyFrame:
    return pl.scan_csv(fsrc, null_values="n/a").with_columns(
        pl.lit(month_name).alias("traded_in")
    )


def preprocess_data(stacked_data: pl.LazyFrame) -> pl.LazyFrame:
    """"""
    return (
        stacked_data.with_columns(pl.col(pl.Utf8).str.strip_chars(" "))
        .drop_nulls("Market Cap")
        .select(
            pl.col("traded_in"),
            "first_name",
            "last_name",
            pl.col("Ticker").alias("ticker"),
            pl.col("Sector").alias("sector"),
            pl.col("Market").alias("market"),
            pl.col("Stock Name").alias("stock_name"),
            parse_str_to_float("Market Cap").alias("market_capitalization"),
            parse_str_to_float("Purchase Price").alias("purchase_price"),
        )
        .with_row_index("id", offset=1)
    )


def parse_str_to_float(column: str) -> pl.Expr:
    """Parse a string to a floating value.

    Examples
    -------
    - "$1.111B" -> 1_111_000
    """
    return pl.col(column).str.extract(r"(\d+.\d+)|(\d+)").cast(pl.Float64) * (
        pl.when(pl.col(column).str.ends_with("K"))
        .then(pl.lit(1_000))
        .when(pl.col(column).str.ends_with("M"))
        .then(pl.lit(1_000_000))
        .when(pl.col(column).str.ends_with("B"))
        .then(pl.lit(1_000_000_000))
        .otherwise(pl.lit(1))
    )


def categorize_data(trade: pl.LazyFrame) -> pl.LazyFrame:
    """Categorise the trading data, specifically the Markey Capitalization
    and Purchase Price.
    """
    categorize_market_capitalization_expr = (
        pl.when(pl.col("market_capitalization") < 100_000_000)
        .then(pl.lit("small"))
        .when(pl.col("market_capitalization") < 1_000_000_000)
        .then(pl.lit("medium"))
        .when(pl.col("market_capitalization") < 100_000_000_000)
        .then(pl.lit("large"))
        .otherwise(pl.lit("huge"))
    )

    categorize_purchase_price_expr = (
        pl.when(pl.col("purchase_price") < 25_000)
        .then(pl.lit("low"))
        .when(pl.col("purchase_price") < 50_000)
        .then(pl.lit("medium"))
        .when(pl.col("purchase_price") < 75_000)
        .then(pl.lit("high"))
        .otherwise(pl.lit("very_high"))
    )

    return trade.with_columns(
        categorize_market_capitalization_expr.alias("market_capitalization_category"),
        categorize_purchase_price_expr.alias("purchase_price_category"),
    )


def rank_data(categorized_trade: pl.LazyFrame) -> pl.LazyFrame:
    """"""
    return categorized_trade.with_columns(
        pl.col("purchase_price")
        .rank(method="random", descending=True)
        .over("market_capitalization_category", "purchase_price_category", "traded_in")
        .alias("rank")
    )


def postprocess_data(ranked_data: pl.LazyFrame) -> pl.LazyFrame:
    return ranked_data.filter(pl.col("rank") <= 5).select(
        "market_capitalization_category",
        "purchase_price_category",
        "traded_in",
        "ticker",
        "sector",
        "market",
        "stock_name",
        pl.col("market_capitalization").round(2),
        "purchase_price",
        pl.col("rank").cast(pl.Int8),
    )


if __name__ == "__main__":
    main()
