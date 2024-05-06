"""2022: Week 8 - Pokémon Evolution Stats

Inputs
------
- __input/2022/input_pkmn_stats_and_evolutions.xlsx

Outputs
-------
- output/2022/wk08_combat_power_analysis.ndjson
"""

import polars as pl


def solve(pokemon_fsrc: str) -> pl.DataFrame:
    """Solve challenge 8 of Preppin' Data 2022.

    Parameters
    ----------
    pokemon_fsrc : str
        Filepath of the input XLSX file containing Pokemon stats and evolutions.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the combat power analysis.

    Notes
    -----
    This function loads the Pokemon stats and evolutions data from the input
    Excel file. It preprocesses the data, including renaming columns where
    necessary. Then, it calculates the combat power for each Pokemon and
    analyzes the evolution stages. Finally, it computes the change in combat
    power from initial to final stages for each Pokemon. The resulting DataFrame
    contains the combat power analysis for Pokemon evolutions.
    """

    # Load the data
    data_dict = load_data(pokemon_fsrc)

    # Unpack and preprocess the data
    pokemon = data_dict["pkmn_stats"]

    pokemon_evolution = data_dict["pkmn_evolutions"].pipe(
        preprocess_pokemon_evolutions_data
    )

    # Collect the output
    combat_power_analysis = view_combat_power_analysis(pokemon, pokemon_evolution)

    return combat_power_analysis


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


def preprocess_pokemon_evolutions_data(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the Pokemon evolutions data.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrames containing the loaded Pokemon evolution data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the preprocessed Pokemon evolution data.
    """

    col_mapper = {col: col.lower() for col in data.columns}

    return data.rename(col_mapper)


def view_combat_power_analysis(
    pokemon: pl.DataFrame, pokemon_evolution: pl.DataFrame
) -> pl.DataFrame:
    """View the combat power analysis for all Pokemon that can experience
    evolution.

    Parameters
    ----------
    pokemon : pl.DataFrame
        DataFrame containing the preprocessed Pokemon data.
    pokemon_evolution : pl.DataFrame
        DataFrame containing the preprocessed Pokemon evolution data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the combat power analysis for Pokemon that
        can experience evolution.
    """

    # Collect the data
    final_stage = view_final_stage(pokemon_evolution)

    combat_power = view_combat_power(pokemon)

    # Expressions
    prop_change_combat_power_expr = pl.col("combat_power_right") / pl.col(
        "combat_power"
    )

    return (
        pokemon_evolution.join(final_stage, on="stage_1")
        .join(combat_power, left_on="stage_1", right_on="name")
        .join(combat_power, left_on="final_stage", right_on="name")
        .join(pokemon, left_on="stage_1", right_on="name")
        .filter(
            (pl.col("final_stage") == pl.col("stage_2"))
            | (pl.col("final_stage") == pl.col("stage_3"))
        )
        .select(
            "stage_1",
            "stage_2",
            "stage_3",
            "pokedex_number",
            "gen_introduced",
            initial_combat_power="combat_power",
            final_combat_power="combat_power_right",
            prop_change_combat_power=prop_change_combat_power_expr,
        )
        .sort("prop_change_combat_power")
    )


def view_final_stage(pokemon_evolution: pl.DataFrame) -> pl.DataFrame:
    """View the final stage for each initial Pokemon species.

    Parameters
    ----------
    pokemon_evolution : pl.DataFrame
        DataFrame containing the preprocessed Pokemon evolution data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the final stage for each initial Pokemon species.
    """

    # Expressions
    final_stage_expr = (
        pl.when(pl.col("stage_3").is_not_null())
        .then("stage_3")
        .when(pl.col("stage_2").is_not_null())
        .then("stage_2")
        .otherwise(pl.lit(None))
    )

    return pokemon_evolution.select("stage_1", final_stage=final_stage_expr).drop_nulls(
        "final_stage"
    )


def view_combat_power(pokemon: pl.DataFrame) -> pl.DataFrame:
    """View each Pokemon's combat power.

    Parameters
    ----------
    pokemon : pl.DataFrame
        DataFrame containing the preprocessed Pokemon data.

    Returns
    -------
    pl.DataFrame
        DataFrame containing each Pokemon's combat power.
    """

    # Expressions
    combat_power_expr = (
        pl.col("hp")
        + pl.col("attack")
        + pl.col("defense")
        + pl.col("special_attack")
        + pl.col("special_defense")
        + pl.col("speed")
    )

    return pokemon.select("name", combat_power=combat_power_expr)
