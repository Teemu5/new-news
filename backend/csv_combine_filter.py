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

def read_with_other_header(
    path: Union[str, Path],
    target_columns: list[str]
) -> pd.DataFrame:
    """
    Read CSV at `path`, skip its own header row, then parse every data row
    padding or truncating to len(target_columns), and return a DataFrame
    with columns=target_columns.
    """
    path = Path(path)
    ncols = len(target_columns)
    records = []

    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # skip the existing (incorrect) header
        try:
            next(reader)
        except StopIteration:
            return pd.DataFrame(columns=target_columns)

        for row in reader:
            # pad short rows
            if len(row) < ncols:
                row = row + ['']*(ncols - len(row))
            # truncate long rows
            elif len(row) > ncols:
                row = row[:ncols]
            records.append(row)

    return pd.DataFrame(records, columns=target_columns)

def process_all(pattern, keep_donor=False):
    input_path = Path('.')
    for csv_file in input_path.glob(pattern):
        outfile = filter_csv_file(csv_file, keep_donor=keep_donor)
        print(f"Processed {csv_file.name} -> {csv_file}")

def combine_csvs(pattern1: str, pattern2: str, output_suffix: str = "_combined.csv") -> list[str]:
    """
    For each file matching pattern1 (e.g. '*_file.csv'):
      – extract key = basename minus the suffix after '*'
      – find a file key + (suffix from pattern2),
      – read both CSVs, append any new columns from the second into the first,
      – write out key + output_suffix.

    Returns list of output file paths.
    """
    suf1 = os.path.basename(pattern1).replace("*", "")
    suf2 = os.path.basename(pattern2).replace("*", "")

    files1 = glob.glob(pattern1)
    files2 = glob.glob(pattern2)
    
    lookup2 = {}
    for f2 in files2:
        name2 = os.path.basename(f2)
        if name2.endswith(suf2):
            key = name2[:-len(suf2)]
            lookup2[key] = f2
    print(f"lookup2:{lookup2}")
    outputs = []
    for f1 in files1:
        name1 = os.path.basename(f1)
        if not name1.endswith(suf1):
            continue
        key = name1[:-len(suf1)]
        f2 = lookup2.get(key)
        if f2 is None:
            continue
        print(f"f2:{f2}")
        try:
            df1 = pd.read_csv(f1)
        except ParserError as e:
            logging.warning(f"Parse error reading {f1}: {e}. Skipping this file.")
            continue

        # read secondary CSV with fallback
        try:
            df2 = read_with_other_header(f2, list(df1.columns))#df2 = pd.read_csv(f2)
        except ParserError as e:
            logging.warning(f"ParserError reading {f2}: {e}. Retrying with python engine, skipping bad lines.")
            try:
                df2 = read_with_other_header(f2, list(df1.columns))#df2 = pd.read_csv(f2, engine='python', on_bad_lines='skip')
            except Exception as e2:
                logging.error(f"Failed again reading {f2}: {e2}. Skipping this pair.")
                continue
        df2 = df2.reindex(columns=df1.columns, fill_value="")

        # 2) Vertically stack (append) df2’s rows under df1
        df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)

        # 3) Write out the combined file
        out_name = f"results/{key}{output_suffix}"
        print(f"write to {out_name}")
        df_combined.to_csv(out_name, index=False)
        outputs.append(out_name)
    return outputs

def main():
    parser = argparse.ArgumentParser(description="Filter and combine partial result CSV files for user-level experiments.")
    parser.add_argument("--filter", type=str, help="Path to a CSV file to filter")
    parser.add_argument("--filter_pattern", type=str, help="pattern to a CSV file to filter")
    parser.add_argument("--pattern", type=str, help="Glob pattern for CSV files to combine")
    parser.add_argument("--pattern1", type=str, help="Glob pattern1 for CSV files to combine with pattern2")
    parser.add_argument("--pattern2", type=str, help="Glob pattern2 for CSV files to combine with pattern1")
    parser.add_argument("--output", type=str, default=None, help="Output filename for combined CSV")
    parser.add_argument("--keep_donor", action="store_true",
                   help="Set to keep lines with donor")
    
    args = parser.parse_args()

    if args.filter:
        print(f"args.keep_donor:{args.keep_donor}")
        filtered_csv = filter_csv_file(args.filter, keep_donor=args.keep_donor)
        print(f"Filtered CSV file: {filtered_csv}")
    if args.pattern:
        combined_csv = combine_csv_files(args.pattern, args.output)
        print(f"Combined CSV file: {combined_csv}")
    if args.pattern1 and args.pattern2:
        csvs =combine_csvs(args.pattern1, args.pattern2)
        print(f"combined:{csvs}")
    if args.filter_pattern:
        process_all(args.filter_pattern, keep_donor=args.keep_donor)

if __name__ == "__main__":
    main()
