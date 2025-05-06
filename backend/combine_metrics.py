#!/usr/bin/env python3

import argparse
import glob
import json
import os
import re
import sys


def collect_metrics(in_dir: str, pattern: str, general_regex=False) -> dict:
    combined = {}
    full_pattern = os.path.join(in_dir, pattern)
    if 'base' != general_regex:
        regex = re.compile(r"^(.*?)_small_valid_small_metrics\.json$")
    else:
        regex = re.compile(r"^(.*?)\.json$")

    for path in glob.glob(full_pattern):
        fname = os.path.basename(path)
        print(f"fname:{fname}")
        match = regex.match(fname)
        if not match:
            continue

        key = match.group(1)
        print(f"key:{key}")
        with open(path, "r", encoding="utf‑8") as fh:
            data = json.load(fh)


        combined[key] = data

    return combined

import json, pandas as pd, numpy as np


def pick4(d, tag):
    return {"model": tag,
            "AUC": d["AUC"],
            "AP":  d.get("AP", np.nan),
            "precision": d["precision"],
            "recall":    d["recall"]}

def vis_plot(COMBINED, META):
    records  = [pick4(v,k) for k,v in json.load(open(COMBINED)).items()]
    records += [pick4(v,k) for k,v in json.load(open(META)).items()]

    df = (pd.DataFrame.from_records(records)
            .set_index("model")
            .sort_values("AUC", ascending=False))
    df.to_csv("four_metric_table.csv")
    import matplotlib.pyplot as plt
    n = len(df)
    x = np.arange(n)
    plt.figure(figsize=(10,4))
    plt.bar(x-0.3, df["AUC"], width=0.3, label="AUC")
    plt.bar(x-0.0, df["precision"],  width=0.3, label="precision")
    plt.bar(x+0.3, df["recall"],  width=0.3, label="recall")
    plt.xticks(x, df.index, rotation=60, ha="right")
    plt.ylabel("Score"); plt.title("AUC, precision and recall of all models")
    plt.legend(); plt.tight_layout()
    plt.savefig("auc_ap_bars.png", dpi=300)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        "--dir", default="."
    )
    parser.add_argument(
        "--pattern",
        default="*_small_valid_small_metrics.json",
    )
    parser.add_argument(
        "--out",
        default="combined_valid_small_metrics.json",
    )
    parser.add_argument(
        "--regex",
        default="base",
    )
    parser.add_argument(
        "--COMBINED",
        default="combined_valid_small_metrics.json",
    )
    parser.add_argument(
        "--META",
        default="combined_meta_valid_small_metrics.json",
    )
    parser.add_argument(
        "--vis",
        default="plot",
    )
    args = parser.parse_args(argv)

### python3 combine_metrics.py --dir meta/results --out combined_meta_valid_small_metrics.json
### python3 combine_metrics.py --dir meta/results --out combined_meta_valid_small_metrics.json --pattern meta_model_XGBClassifier_hist_300_*_small_train_small_valid.json --regex true
###  python3 combine_metrics.py --vis plot --COMBINED combined_valid_small_metrics.json --META combined_meta_valid_small_metrics.json
    
    if args.vis == "plot":
        vis_plot(args.COMBINED, args.META)
        return
    metrics = collect_metrics(args.dir, args.pattern, args.regex)
    if not metrics:
        print("No files matched", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", encoding="utf‑8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Wrote {len(metrics)} metric dicts to {args.out}")


if __name__ == "__main__":
    main()
