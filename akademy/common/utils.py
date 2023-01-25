import logging
import os
from typing import List

import numpy as np
import pandas as pd
from akademy.common import project_paths


def format_float(value: float, do_rounding: bool = True) -> str:
    """
    Convenience function to adjust report checkpoint_save_dir formatting.
    Turns floating point numbers into comma-separated strings
    rounded to the nearest 2 decimal places.
    """
    if do_rounding:
        value = round(value, 2)
    return f"{value:,}"


def get_file_extension(filepath: str) -> str:
    """
    Given the path to a file, return only the extension such that the following
    cases are handled:
        1. file.ext -> ext
        2. file.tar.gz -> tar.gz
        3. file -> ''
    """
    filename = os.path.basename(filepath)
    if "." not in filename:
        return filename
    return ".".join(filename.split('.')[1:])


def remove_old_file_versions(
        filepath: str,
        remove_key: str,
        keep_key: str = None,
        keep_count: int = 0
) -> List[str]:
    """
    Given an absolute filepath to a file, removes all files within that directory
    that contain the <key> value within them, keeping <keep> many files,
    where the deleted files are those oldest via last_mod check.
    Args:
        filepath: path to the file from which to remove old versions.
        keep_count: the number of files to keep. Default 0 means delete them all.
        remove_key: character sequence to identify similar files.
            e.g. if <key> in <file>
        keep_key: files matching <keep_key> in <file> are not considered.
    Returns:
        List of deleted filepaths.
    """
    file_dir = os.path.abspath(os.path.dirname(filepath))
    file_name = os.path.basename(file_dir).split('.')[0]
    file_ext = ".".join(os.path.basename(filepath).split('.')[1:])

    # get list of filenames intended for removal.
    files = [f for f in os.listdir(file_dir) if remove_key in f]

    # remove files matching keep mask
    if remove_key:
        files = [f for f in files if keep_key not in f]

    # sort by filemod time
    files = sorted(files,
                   key=lambda x: os.path.getmtime(
                       os.path.abspath(os.path.join(file_dir, x))
                   ),
                   reverse=True
                   )

    # keep max if specified
    if keep_count > 0:
        if keep_count > len(files):
            pass
        else:
            files = files[keep_count:]

    # delete the older files
    deleted = []
    for file in files:

        fullpath = os.path.abspath(os.path.join(
            file_dir,
            file
        ))
        deleted.append(fullpath)
        logging.info(f'removing file: {fullpath}')
        os.remove(fullpath)

    return deleted


def load_csv_ohlcv_data(
        filepath: str,
        count: int = 0,
        date_ms_unit: str = "ns") -> pd.DataFrame:
    """
    Loads financial data presumed to have the following columns present:
        Open, High, Low, Close, Volume, Date
    Args:
        filepath: the path to the CSV file to open.
        count: total number of records to return; earlier rows dropped; 0 == all
        date_ms_unit: the unit measure of the date timestamp. Default Nanoseconds.
    Return:
        data as a DataFrame with open, high, low, close, volume
        headers and indexed by ascending date
    """
    data = pd.read_csv(filepath)

    # lowercase all column names for standardization
    data.columns = [x.lower() for x in data.columns]

    # drop all but essential pricing + date + volume
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]

    # convert types
    data = data.astype(
        {
            'date': f'datetime64[{date_ms_unit}]',
            'open': np.float32,
            'high': np.float32,
            'low': np.float32,
            'close': np.float32,
            'volume': np.float32
        }
    )

    # reindex
    data.set_index('date', inplace=True)

    # 0 == all the data
    if count == 0:
        return data

    # check more data not asked for than available
    if count > len(data):
        raise Exception(
            f"The request number of records <{count}> "
            f"is more than total available: <{len(data)}>"
        )

    # otherwise, return the <count> most recent data items
    return data[len(data) - count:]


def load_spy_daily(count: int = 0):
    """
    Loads the locally-saved SPY pricing data with daily OHLCV prices
    ranging from 1/29/1993 to 9/6/2022 in ascending order.
    7455 total entries in daily increments.
    Arguments:
        count: number of days to return from the max period. End is most recent.
    Notes:
        'close' is replaced by 'Adj Close'; but renamed as 'close'
    Returns:
        pd.DataFrame object
    """
    return load_csv_ohlcv_data(project_paths.SPY_DATA, count=count)


def minmax_normalize(data: np.ndarray) -> np.ndarray:
    """
    Converts all values within an ndarray to values within the range of 0-1
    where
    Args:
        data: numpy array of data.
    Returns:
        a new numpy array containing values in range 0-1
    """
    _min = np.min(data)
    _ptp = np.ptp(data)
    return (data-_min)/_ptp
