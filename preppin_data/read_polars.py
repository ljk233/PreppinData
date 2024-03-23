"""Module for reading data into Polars DataFrames."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from zipfile import ZipFile

import polars as pl


"""Module for reading data into Polars DataFrames."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from zipfile import ZipFile

import polars as pl


def read_zip_file(zip_file: str) -> dict[str, pl.DataFrame]:
    """Extract data from a ZIP file and ingest it into a dictionary of Polars DataFrames.

    Parameters
    ----------
    zip_file : str
        The path to the ZIP file containing the data.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary mapping file names to Polars DataFrames containing the extracted data.
    """
    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        with ZipFile(zip_file) as zf:
            zf.extractall(tmp_dir_path)

        return read_directory(tmp_dir_path)


def read_directory(directory: Path) -> dict[str, pl.DataFrame]:
    """Collect data files from a directory and ingest them into a dictionary of Polars DataFrames.

    Parameters
    ----------
    directory : Path
        The path to the directory containing the data files.

    Returns
    -------
    dict[str, pl.DataFrame]
        A dictionary mapping file names to Polars DataFrames containing the ingested data.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"FAILED: Directory {directory} does not exist.")

    data_dict = {}
    for file_path in directory.iterdir():
        if file_path.is_file():
            data_dict[file_path.name] = read_file(file_path)

    return data_dict


def read_file(file_path: Path) -> pl.DataFrame:
    """Read data from a file into a Polars DataFrame.

    Parameters
    ----------
    file_path : Path
        The path to the file to be read.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the data read from the file.

    Notes
    -----
    This function dynamically selects the appropriate Polars read function
    based on the file extension and attempts to read the file.
    If the file encoding is not utf8, then it falls back to using the
    'utf8-lossy' encoding.

    Examples
    --------
    >>> file_path = Path('data.csv')
    >>> df = read_file(file_path)
    >>> df.shape()
    (1000, 5)
    """
    read_fn = get_polars_read_strategy(file_path.suffix)

    try:
        return read_fn(file_path)
    except pl.ComputeError:
        return read_fn(file_path, encoding="utf8-lossy")


def get_polars_read_strategy(suffix: str) -> Callable[..., pl.DataFrame]:
    """Return the appropriate Polars read function based on the file suffix.

    Parameters
    ----------
    suffix : str
        The file suffix indicating the file format.

    Returns
    -------
    Callable[..., pl.DataFrame]
        The appropriate Polars read function for the given file format.

    Raises
    ------
    ValueError
        If the file suffix is not supported.
    """
    lsuffix = suffix.lower().lstrip(".")
    match lsuffix:
        case "csv":
            return pl.read_csv
        case "xlsx":
            return pl.read_excel
        case _:
            raise ValueError(f"File suffix {suffix} is not supported.")
