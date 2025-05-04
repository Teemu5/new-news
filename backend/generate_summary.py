#!/usr/bin/env python3
import argparse
import pandas as pd
from model_utils import compute_experiment_summary
from pathlib import Path
import matplotlib.pyplot as plt

def per_file_summaries(pattern: str, k_values=[5,10,20,50,100], auc=False):
    rows = []
    for csv_path in Path('.').glob(pattern):
        _, summary_dict = compute_experiment_summary(str(csv_path), k_values=k_values, auc=auc)
        summary_dict["file"] = csv_path.name
        summary_dict["method"] = summary_dict["file"].split("_")[1]
        rows.append(summary_dict)
    df = pd.DataFrame(rows).set_index("file")
    df = df.sort_index() # sort based on files
    return df

def main():
    parser = argparse.ArgumentParser(description="Compute summary statistics for user-level experiment CSV files.")
    parser.add_argument("--pattern", type=str, help="Glob pattern for CSV files to combine (e.g., 'bagging_user_level_partial_results_*_*.csv').", required=True)
    parser.add_argument("--k_values", type=str, default="5,10,20,50,100",
                        help="Comma-separated list of k values (default: 5,10,20,50,100)")
    parser.add_argument("--per_file", action="store_true", help="Produce one-row-per-file summary instead of global.")
    parser.add_argument("--auc", action="store_true", help="Include auc column")
    args = parser.parse_args()
    
    k_values = [int(k) for k in args.k_values.split(',')]
        
    if args.per_file:
        summary_by_file = per_file_summaries(args.pattern, k_values, args.auc)
        suffix = args.pattern.split('*')[1]
        outfile = f"per_file_summary{suffix}"
        print(f"writing to {outfile}")
        summary_by_file.to_csv(outfile)
    else:
        summary_df, summary_dict = compute_experiment_summary(args.pattern, k_values=k_values)
        
        print("Summary statistics:")
        print(summary_df)

        output_file = f"{args.pattern}_summary.csv"
        summary_df.to_csv(output_file, index=True)
        print(f"Summary CSV saved to {output_file}")

if __name__ == "__main__":
    main()
