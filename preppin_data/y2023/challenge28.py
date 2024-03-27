"""2023: Week 28 - Prep School Track Team

See solution output at "output/2023/wk28_top_athletes.ndjson".
"""

import polars as pl


def solve(input_fsrc: str) -> pl.DataFrame:
    """Solve challenge 28 of Preppin' Data 2023."""
    # Load the data
    data_dict = pl.read_excel(input_fsrc, sheet_id=0)

    # Preprocess the data
    pre_student = data_dict["Students"]
    pre_track_time = data_dict["Track Times"]
    pre_benchmark = preprocess_benchmark(data_dict["Benchmarks"])

    # Select the track team
    is_under_benchmark_time_pred = pl.col("time") <= pl.col("benchmark_time")
    is_200m_event_pred = pl.col("track_event") == "200m"
    is_under_25secs_pred = pl.col("time") < 25

    return (
        pre_track_time.join(pre_student, on="id")
        .join(pre_benchmark, on=["track_event", "gender", "age"])
        .filter(
            is_under_benchmark_time_pred
            & (is_200m_event_pred & is_under_25secs_pred).not_()
        )
        .with_columns(
            pl.col("time")
            .rank("min")
            .over("gender", "age", "track_event")
            .alias("student_rank")
        )
    )


def preprocess_benchmark(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocess the benchmark data."""
    renamer_dict = {
        "Gender ": "gender",
        "Age ": "age",
        "Event ": "track_event",
        "Benchmark": "benchmark_time",
    }

    return data.rename(renamer_dict)
