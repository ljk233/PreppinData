"""2023.challenge14
"""

import polars as pl

from ..read_polars import read_zip_file


def solve(wits_zfsrc: str, country_code_fsrc: str) -> pl.DataFrame:
    """Solve challenge 14 of Preppin' Data 2023.

    Notes
    -----
    See solution output at output/2023/wk14_world_imports_exports.ndjson.
    """
    # Preprocess the data
    pre_wits = collect_preprocessed_wits_data(wits_zfsrc)
    country_geo = preprocess_country_geo(country_code_fsrc)

    # Filter and pivot the WITS data on Indicator
    transformed_wits = pre_wits.pipe(transform_wits)

    return (
        transformed_wits
        # Combine the data
        .join(
            country_geo,
            left_on="reporting_country_code",
            right_on="iso3_code",
            how="left",
        )
        .join(
            country_geo,
            left_on="partner_country_name",
            right_on="label_en",
            how="left",
        )
        .select(
            "year",
            "reporting_country_name",
            "reporting_country_code",
            pl.col("latitude").alias("reporting_country_latitude"),
            pl.col("longitude").alias("reporting_country_longitude"),
            "partner_country_name",
            pl.col("iso3_code").alias("partner_country_iso3_code"),
            pl.col("latitude_right").alias("partner_country_latitude"),
            pl.col("longitude_right").alias("partner_country_longitude"),
            "product_category",
            "indicator_type",
            "top5_export_partner_trade",
            "top5_import_partner_trade",
            "pct_share_top5_export_partners",
            "pct_share_top5_import_partners",
        )
    )


def collect_preprocessed_wits_data(wits_zip_fsrc: str) -> pl.DataFrame:
    """Gather all the data in the source zip file into a single DataFrame
    that is ready for analysis.
    """
    data_dict = read_zip_file(wits_zip_fsrc)

    pre_data_dict = {f: data.pipe(preprocess_wits) for f, data in data_dict.items()}

    # Append the three-letter code to the WITS data
    data_arr = [
        pre_data.pipe(append_three_letter_country_code, f)
        for f, pre_data in pre_data_dict.items()
    ]

    # Stack the preprocessed WITS data
    return pl.concat(data_arr)


def preprocess_wits(wits_data: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the WITS data."""
    # Melting identity variables
    id_vars = [
        "Reporter",
        "Partner",
        "Product categories",
        "Indicator Type",
        "Indicator",
    ]

    # Renamer dict
    renamer_dict = {
        "Reporter": "reporting_country_name",
        "Partner": "partner_country_name",
        "Product categories": "product_category",
        "Indicator Type": "indicator_type",
        "Indicator": "indicator",
    }

    return (
        wits_data.with_columns(pl.col(pl.Utf8).str.strip_chars(" "))
        .melt(id_vars=id_vars, variable_name="year")
        .drop_nulls()
        .with_columns(pl.col("year").cast(pl.Int16), pl.col("value").cast(pl.Float64))
        .rename(renamer_dict)
    )


def append_three_letter_country_code(
    wits_data: pl.DataFrame,
    fsrc: str,
    col_alias: str = "reporting_country_code",
) -> pl.DataFrame:
    """Append the three-letter country code held in the file name to the
    WITS data.
    """
    return wits_data.with_columns(pl.lit(fsrc[3:6]).alias(col_alias))


def preprocess_country_geo(country_code_fsrc: str) -> pl.DataFrame:
    """Preprocess the country codes data.

    Notes
    -----
    For brevity, we only preprocess the columns needed for this analysis,
    """
    return (
        pl.read_csv(country_code_fsrc, separator=";")
        .with_columns(
            pl.col("geo_point_2d").str.split(",").alias("geo_point_2d_arr"),
        )
        .select(
            pl.col("ISO3 CODE").alias("iso3_code"),
            pl.col("LABEL EN").alias("label_en"),
            pl.col("geo_point_2d_arr").list[0].cast(pl.Float64).alias("latitude"),
            pl.col("geo_point_2d_arr").list[1].cast(pl.Float64).alias("longitude"),
        )
    )


def transform_wits(pre_data: pl.DataFrame) -> pl.DataFrame:
    """Filter and pivot the data on and pivot the WITS data."""
    # Predictate expressions
    is_import_export_pre = pl.col("indicator_type").is_in(["Import", "Export"])
    is_valid_reporter_pred = (
        pl.col("reporting_country_name")
        .is_in(["World", "European Union", "Occ.Pal.Terr", "Other Asia, nes"])
        .not_()
    )
    is_valid_parter_pred = (
        pl.col("partner_country_name")
        .is_in(["World", "...", "Special Category"])
        .not_()
    )

    # Renamer dict
    renamer_dict = {
        "Trade (US$ Mil)-Top 5 Export Partner": "top5_export_partner_trade",
        "Trade (US$ Mil)-Top 5 Import Partner": "top5_import_partner_trade",
        "Partner share(%)-Top 5 Export Partner": "pct_share_top5_export_partners",
        "Partner share(%)-Top 5 Import Partner": "pct_share_top5_import_partners",
    }

    # ID variables for pivoting
    id_vars = [
        "reporting_country_name",
        "reporting_country_code",
        "partner_country_name",
        "product_category",
        "indicator_type",
        "year",
    ]

    return (
        pre_data.filter(
            is_import_export_pre & is_valid_reporter_pred & is_valid_parter_pred
        )
        .pivot(values="value", columns="indicator", index=id_vars)
        .rename(renamer_dict)
    )
