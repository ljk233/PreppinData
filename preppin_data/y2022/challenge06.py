"""2022: Week 6 - [Seven] letter Scrabble Words

Note, 

Inputs
------
- __input/2022/7 letter words.xlsx

Outputs
-------
- output/2022/wk06_seven_letter_word_analysis.ndjson

Notes
-----
We set the float precision to suppress Polar's use of scientific notation.
"""

import polars as pl


pl.Config.set_float_precision(14)


def solve(seven_letter_word_fsrc: str) -> pl.DataFrame:
    """Solve challenge 6 of Preppin' Data 2022.

    Parameters
    ----------
    seven_letter_word_fsrc : str
        Filepath of the input XLSX file for Week 6.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the analysis of seven-letter words.

    Notes
    -----
    This function loads data from the input Excel file, preprocesses it,
    performs the required analysis, and returns a DataFrame with the results.
    """

    # Load the data
    data_dict = load_data(seven_letter_word_fsrc)

    # Preprocess the data
    word = data_dict["7 letter words"].pipe(preprocess_words_data)

    scrabble_score = data_dict["Scrabble Scores"].pipe(preprocess_scrabble_scores_data)

    # Collect the output
    seven_letter_word_analysis = view_seven_letter_word_analysis(word, scrabble_score)

    return seven_letter_word_analysis


def load_data(seven_letter_word_fsrc: str) -> dict[str, pl.DataFrame]:
    """Load data from the input Excel file.

    Parameters
    ----------
    seven_letter_word_fsrc : str
        Filepath of the input XLSX file for Week 6.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary containing Polars DataFrames with loaded data.
        The keys are the names of the sheets in the Excel file,
        and the values are the corresponding DataFrames.
    """

    data_dict = pl.read_excel(seven_letter_word_fsrc, sheet_id=0)

    return data_dict


def preprocess_words_data(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the seven letter words data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing seven-letter words data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed seven-letter words data.
    """

    return data.select(word=pl.col("7 letter word").str.to_uppercase())


def preprocess_scrabble_scores_data(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the scrabble scores data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing scrabble scores data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed scrabble scores data.
    """

    return data.pipe(clean_scrabble_scores_data)


def clean_scrabble_scores_data(data: pl.DataFrame) -> pl.DataFrame:
    """Clean the scrabble scores data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing scrabble scores data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the cleansed scrabble scores data.
    """

    # Expressions
    scrabble = pl.col("Scrabble").str.replace_all("×", "")

    points_patt = r"^(\d+)"

    points_expr = scrabble.str.extract(points_patt).cast(pl.Int64)

    tiles_patt = r"(\w+)\s(\d+)"

    tiles_expr = scrabble.str.extract_all(tiles_patt)

    clean_tiles_expr = pl.col("tiles").str.replace("Blank", "_")

    letter_expr = clean_tiles_expr.str.extract(r"([A-Z])")

    frequency_expr = clean_tiles_expr.str.extract(r"(\d+)").cast(pl.Int64)

    return (
        data.with_columns(points=points_expr, tiles=tiles_expr)
        .explode("tiles")
        .select(
            "points",
            letter=letter_expr,
            frequency=frequency_expr,
        )
    )


def view_seven_letter_word_analysis(
    word: pl.DataFrame, scrabble_score: pl.DataFrame
) -> pl.DataFrame:
    """View the analysis of seven-letter words.

    Parameters
    ----------
    word : pl.DataFrame
        DataFrame containing the preprocessed seven-letter words data.
    scrabble_score : pl.DataFrame
        DataFrame containing the preprocessed scrabble scores data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the analysis of seven-letter words.
    """

    # Collect the data
    word_probability = view_word_probability(word, scrabble_score)

    total_points_per_word = view_total_points_per_word(word, scrabble_score)

    return (
        word.join(word_probability, on="word")
        .join(total_points_per_word, on="word")
        .filter(pl.col("probability") > 0)
        .with_columns(
            probability_rank=pl.col("probability").rank("min", descending=True),
            total_points_rank=pl.col("total_points").rank("min", descending=True),
        )
    )


def view_word_probability(
    word: pl.DataFrame, scrabble_score: pl.DataFrame
) -> pl.DataFrame:
    """View the probability of drawing all the tiles necessary to create
    each word.

    Parameters
    ----------
    word : pl.DataFrame
        DataFrame containing the preprocessed seven-letter words data.
    scrabble_score : pl.DataFrame
        DataFrame containing the preprocessed scrabble scores data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing te probability of drawing all titles needed
        to create a word.
    """

    # Collect the data
    letter_probability = view_letter_probability(scrabble_score)

    agg_letters_per_word = aggregate_letters_per_word(word)

    # Expressions
    partial_probability_expr = (
        pl.when(pl.col("word_frequency") <= pl.col("frequency"))
        .then(pl.col("probability").pow("word_frequency"))
        .otherwise(pl.lit(0))
    )

    return (
        agg_letters_per_word.join(scrabble_score, on="letter")
        .join(letter_probability, on="letter")
        .with_columns(partial_probability=partial_probability_expr)
        .group_by("word")
        .agg(probability=pl.col("partial_probability").product())
    )


def view_letter_probability(scrabble_score: pl.DataFrame) -> pl.DataFrame:
    """View the probability of a letter tile being drawn.

    Parameters
    ----------
    scrabble_score : pl.DataFrame
        DataFrame containing the preprocessed scrabble scores data.
    """

    frequency = pl.col("frequency")

    num_tiles_expr = frequency.sum()

    probability_expr = frequency / num_tiles_expr

    return scrabble_score.select("letter", probability=probability_expr)


def aggregate_letters_per_word(word: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the frequency of each letter in a word.

    Parameters
    ----------
    word : pl.DataFrame
        DataFrame containing the preprocessed seven-letter words data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the probability of a letter tile being drawn.
    """

    return (
        word.with_columns(letter=pl.col("word").str.split(by=""))
        .explode("letter")
        .group_by("word", "letter")
        .agg(word_frequency=pl.len())
        .filter(pl.col("letter") != "")
    )


def view_total_points_per_word(
    word: pl.DataFrame, scrabble_score: pl.DataFrame
) -> pl.DataFrame:
    """View the total points each word is worth.

    Parameters
    ----------
    word : pl.DataFrame
        DataFrame containing the preprocessed seven-letter words data.
    scrabble_score : pl.DataFrame
        DataFrame containing the preprocessed scrabble scores data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the total points each word is worth.
    """

    # Collect the data
    agg_letters_per_word = aggregate_letters_per_word(word)

    # Expressions
    partial_total_points_expr = pl.col("word_frequency") * pl.col("points")

    return (
        agg_letters_per_word.join(scrabble_score, on="letter")
        .with_columns(partial_total_points=partial_total_points_expr)
        .group_by("word")
        .agg(total_points=pl.sum("partial_total_points"))
    )
