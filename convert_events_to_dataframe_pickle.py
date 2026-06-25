#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Code to ingest multiple R data files (RDS) into Python objects and save as
a single pickle file for later use in tensor decomposition and plotting."""

import sys
import warnings
import rdata
from pathlib import Path
from multiprocessing import Pool
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

# Suppress UserWarning about missing constructors from rdata
warnings.filterwarnings("ignore", category=UserWarning, module="rdata")


def read_rds_file(file_path):
    """Read an RDS file and return the contained object."""
    try:
        data = rdata.read_rds(file_path)
        return data
    except Exception as e:
        print(f"Error reading RDS file {file_path}: {e}")
        return None


if __name__ == "__main__":
    event_files = sorted(list(Path(sys.argv[1]).glob("*")))
    print(f"Found {len(event_files)} files to process")

    # Use multiprocessing to read files in parallel
    n_workers = os.cpu_count() or 1
    print(f"Using {n_workers} worker processes")

    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(read_rds_file, event_files),
                total=len(event_files),
                desc="Reading RDS files",
            )
        )

    # Filter out None results from failed reads
    valid_data = [r for r in results if r is not None]
    print(f"Successfully read {len(valid_data)} files")

    if valid_data:
        # Concatenate all data
        print("Concatenating data...")
        event_data = pd.concat(valid_data, ignore_index=True)

        # Replace -1 with NaN (missing values)
        print("Replacing -1 with NaN...")
        event_data = event_data.replace(-1, np.nan)

        print(f"Saving combined data with {len(event_data)} rows...")
        event_data.to_pickle("combined_event_data.pkl")
        print("Done!")
