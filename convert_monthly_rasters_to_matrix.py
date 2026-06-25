#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import numpy as np
from numpy.lib.format import open_memmap
import tifffile as tiff
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime
from tqdm import tqdm


def load_tiff(path) -> tuple[np.ndarray, datetime]:
    # Suppress the tifffile log warnings
    logging.getLogger("tifffile").disabled = True

    try:
        img = tiff.imread(path)
    except Exception as e:
        print(f"Failed on {path}: {e}")
        return np.eye(2), datetime.now()

    # Data labelled as -1 actually mean "measured but not burnt", which is a
    # severity score of 0.
    # Data labelled as N/A or NaN are regions outside our field of interest,
    # so we ecode as -1 (not a valid metric score)
    img = np.where(img == -1, 0, img)
    img = np.where(img < -1, -1, img)

    # Get the time-stamp from the filename assuming it is formatted like:
    # severity_YYYY-MM-DD.tif
    timestamp = Path(path).stem.split("_")[-1]
    time = datetime.strptime(timestamp, "%Y-%m-%d")

    return img, time


def construct_severity_matrix(files, to_dtype=np.int8):
    """Use memory mapping instead of storing full array in RAM"""
    # First pass: get dimensions so we can construct the mem-mapped file
    first_img, _ = load_tiff(files[0])

    # Create memory-mapped file
    severity = open_memmap(
        "severity.npy",
        dtype=to_dtype,
        mode="w+",
        shape=(len(files), *first_img.shape),
    )
    times = np.empty(len(files), dtype="datetime64[D]")

    # Load and save one at a time or in batches
    with Pool() as pool:
        for i, (img, time) in enumerate(
            tqdm(
                pool.imap(load_tiff, files),
                total=len(files),
                desc="Loading TIF files",
            )
        ):
            severity[i] = img.astype(to_dtype)
            times[i] = np.datetime64(time, "D")

    # Also dump the time stamps as a separate file
    np.save("timestamps.npy", times)

    # Clean up the mem-mapped file
    severity.flush()
    del severity


if __name__ == "__main__":
    folder = Path("/data/bmeyers/DBCA/data/monthly_severity_grids")
    files = sorted(folder.glob("*.tif"))

    odname = Path("severity_rasters_i8.npy")
    otname = Path("severity_rasters_timestamps.npy")
    if not odname.exists() or not otname.exists():
        print("Constructing severity matrix...")
        construct_severity_matrix(files, to_dtype=np.int8)

