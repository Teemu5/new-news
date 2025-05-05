#!/usr/bin/env python3
import argparse
from model_utils import filter_csv_file, combine_csv_files
from pathlib import Path
import os
import glob
import pandas as pd

import csv
import pandas as pd
from pathlib import Path
from typing import Union

def process_all(pattern, keep_donor=False):
    input_path = Path('.')
    for csv_file in input_path.glob(pattern):
        outfile = filter_csv_file(csv_file, keep_donor=keep_donor)
        print(f"Processed {csv_file.name} -> {csv_file}")

def main():
    parser = argparse.ArgumentParser(description="Filter and combine partial result CSV files for user-level experiments.")
    parser.add_argument("--filter", type=str, help="Path to a CSV file to filter")
    parser.add_argument("--filter_pattern", type=str, help="pattern to a CSV file to filter")
    parser.add_argument("--keep_donor", action="store_true",
                   help="Set to keep lines with donor")
    
    args = parser.parse_args()

    if args.filter:
        print(f"args.keep_donor:{args.keep_donor}")
        filtered_csv = filter_csv_file(args.filter, keep_donor=args.keep_donor)
        print(f"Filtered CSV file: {filtered_csv}")
    if args.filter_pattern:
        process_all(args.filter_pattern, keep_donor=args.keep_donor)

if __name__ == "__main__":
    main()
