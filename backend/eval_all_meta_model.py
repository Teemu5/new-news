from model_utils import evaluate_meta_from_parquet
import argparse, pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NRS evaluation for specified cluster(s).")
    parser.add_argument("--dataset", type=str, default="valid", help="Dataset type: train or valid")
    parser.add_argument("--train_dataset", type=str, default="valid", help="Dataset type meta model was trained on: train or valid")
    parser.add_argument("--model_type", type=str, default="every", help="model type: cluster or category")
    parser.add_argument("--load_best_model", action='store_true', help="Load best model")
    parser.add_argument("--cluster_id", type=str, default=None,
                        help="Cluster ID (or comma-separated list) to evaluate (if not provided, all clusters will be processed)")
    parser.add_argument("--dataset_size", type=str, default="large", help="Dataset size: large or small")
    parser.add_argument("--train_dataset_size", type=str, default="small", help="Training dataset size: large or small")
    parser.add_argument("--model_size", type=str, default="large", help="Model size: large or small")
    parser.add_argument("--dont_resume", action='store_true', help="dont resume from cp")
    parser.add_argument("--process_dfs", action='store_true', help="Process dataframes if needed")
    parser.add_argument("--process_behaviors", action='store_true', help="Process behaviors if needed")
    parser.add_argument("--retrain", type=str, default="none", help="comma separate list of model types to retrain")
    parser.add_argument("--epochs", type=int, default=1, help="epochs number of (global) model to train/load")
    parser.add_argument("--meta_model_type", type=str, default="XGBClassifier", help="meta model type LogisticRegression, LogisticRegressionCV, XGBClassifier or SGDClassifier")
    parser.add_argument("--booster", type=str, default="default", help="booster for XGBClassifier")
    parser.add_argument("--tree_method", type=str, default="hist", help="tree_method for XGBClassifier")
    parser.add_argument("--n_estimators", type=int, default=300, help="n_estimators for XGBClassifier")

    parser.add_argument("--parquet_dir", type=str, default="base_preds", help="parquet_dir for base preds")
    parser.add_argument("--estimator", type=str, default="lbfgs", help="estimator for meta model")
    parser.add_argument("--col", type=str, default="Hist_full", help="e.g. Hist_full, defines dataset_type_suffix")
    parser.add_argument("--v", type=int, default=0, help="version")
    parser.add_argument("--eval_frac", type=float, default=0.04, help="eval_fraction used")
    parser.add_argument("--test_size", type=float, default=-0.1, help="test_size which validation scores meta model is trained on. Ignored when given negative value")
    parser.add_argument("--pivots", type=str, default="", help="pivots in comma separated list")
    parser.add_argument("--ks", type=str, default="", help="ks in comma separated list")
    parser.add_argument("--model_arc_type", type=str, default="fastformer", help="model_arc_type")
    parser.add_argument("--timed", action='store_true', help="timed")
    parser.add_argument("--use_cpu", action='store_true', help="use_cpu")


    args = parser.parse_args()
    pivots = [int(n) for n in args.pivots.split(',')] if args.pivots != "" else []
    ks = [int(n) for n in args.ks.split(',')] if args.ks != "" else []
    vers_suffix = ""
    if args.v > 1:
        vers_suffix = f"_v{args.v}"


    dataset_type_suffixes = []
    dataset_type_suffix = ""
    if args.col != "":
        dataset_type_suffix = f"_{args.col}"
        dataset_type_suffixes.append(dataset_type_suffix)

    for pivot in pivots:
        dataset_type_suffix = f"_Hist_swap{pivot}"
        dataset_type_suffixes.append(dataset_type_suffix)
    for k in ks:
        dataset_type_suffix = f"_Hist_k{k}"
        dataset_type_suffixes.append(dataset_type_suffix)


    eval_frac_suffix = ""
    if args.eval_frac < 1.0:
        eval_frac_suffix = f"_{args.eval_frac}"

    meta_name = args.meta_model_type
    if args.tree_method != '':
        meta_name = f"{meta_name}_{args.tree_method}"
    if args.booster != 'default':
        meta_name = f"{meta_name}_{args.booster}"
    model_arc_type_prefix=""
    if args.model_arc_type != "":
        model_arc_type_prefix=f"{args.model_arc_type}_"
    if args.timed:
        vers_suffix = f"{vers_suffix}_timed"
    if args.use_cpu:
        vers_suffix = f"{vers_suffix}_cpu"
    
    if args.model_type == "every":
        model_type = "all"
    else:
        model_type=args.model_type
    for dataset_type_suffix in dataset_type_suffixes:
        pattern = f"{model_arc_type_prefix}*_{args.model_size}_{args.dataset}_{args.dataset_size}_test_size_{args.test_size}{vers_suffix}{dataset_type_suffix}{eval_frac_suffix}.parquet"
        #pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}.parquet"
        meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{args.model_type}_{args.model_size}_{args.train_dataset}_{args.train_dataset_size}_test_size_{args.test_size}{vers_suffix}"
        #if args.test_size < 0:
        #    meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{args.model_type}_{args.model_size}_{args.train_dataset}_{args.train_dataset_size}"
        #    pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}.parquet"
        meta_path = f"meta/{meta_model_base}.joblib"
        print(f"meta_path:{meta_path}")
        store_metrics_path = f"meta/results/{meta_model_base}{dataset_type_suffix}_{args.dataset}.json"
        store_parquet_path = f"meta/{meta_model_base}{dataset_type_suffix}_{args.dataset}.parquet"
        path = pathlib.Path(store_metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        evaluate_meta_from_parquet(
            meta_path=meta_path,
            parquet_dir=args.parquet_dir,
            pattern=pattern,
            store_parquet_path=store_parquet_path,
            store_metrics_path=store_metrics_path,
            base_model_type=args.model_type,
            booster=args.booster)


    model_type = "cluster"
    for dataset_type_suffix in dataset_type_suffixes:
        pattern = f"{model_arc_type_prefix}*_{args.model_size}_{args.dataset}_{args.dataset_size}_test_size_{args.test_size}{vers_suffix}{dataset_type_suffix}{eval_frac_suffix}.parquet"
        #pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}.parquet"
        meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{args.model_type}_{args.model_size}_{args.train_dataset}_{args.train_dataset_size}_test_size_{args.test_size}{vers_suffix}"
        #if args.test_size < 0:
        #    meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{args.model_type}_{args.model_size}_{args.train_dataset}_{args.train_dataset_size}"
        #    pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}.parquet"
        meta_path = f"meta/{meta_model_base}.joblib"
        print(f"meta_path:{meta_path}")
        store_metrics_path = f"meta/results/{meta_model_base}{dataset_type_suffix}_{args.dataset}.json"
        store_parquet_path = f"meta/{meta_model_base}{dataset_type_suffix}_{args.dataset}.parquet"
        path = pathlib.Path(store_metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        evaluate_meta_from_parquet(
            meta_path=meta_path,
            parquet_dir=args.parquet_dir,
            pattern=pattern,
            store_parquet_path=store_parquet_path,
            store_metrics_path=store_metrics_path,
            base_model_type=args.model_type,
            booster=args.booster)


    model_type = "category"
    for dataset_type_suffix in dataset_type_suffixes:
        pattern = f"{model_arc_type_prefix}*_{args.model_size}_{args.dataset}_{args.dataset_size}_test_size_{args.test_size}{vers_suffix}{dataset_type_suffix}{eval_frac_suffix}.parquet"
        #pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}.parquet"
        meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{args.model_type}_{args.model_size}_{args.train_dataset}_{args.train_dataset_size}_test_size_{args.test_size}{vers_suffix}"
        #if args.test_size < 0:
        #    meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{args.model_type}_{args.model_size}_{args.train_dataset}_{args.train_dataset_size}"
        #    pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}.parquet"
        meta_path = f"meta/{meta_model_base}.joblib"
        print(f"meta_path:{meta_path}")
        store_metrics_path = f"meta/results/{meta_model_base}{dataset_type_suffix}_{args.dataset}.json"
        store_parquet_path = f"meta/{meta_model_base}{dataset_type_suffix}_{args.dataset}.parquet"
        path = pathlib.Path(store_metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        evaluate_meta_from_parquet(
            meta_path=meta_path,
            parquet_dir=args.parquet_dir,
            pattern=pattern,
            store_parquet_path=store_parquet_path,
            store_metrics_path=store_metrics_path,
            base_model_type=args.model_type,
            booster=args.booster)


    
# TRAIN DEFAULT MODELS WITH COMMAND: python3 train_all_models.py
# EVAL DEFAULT MODEL WITH COMMAND: python3 eval_all_models.py
# TRAIN DEFAULT META MODELS WITH COMMAND: python3 train_all_meta_models.py
# EVAL DEFAULT META MODELS WITH COMMAND: python3 eval_all_meta_models.py

# Meta models' results are found at backend/meta/results/ in .json files
# Base models' results are found at backend/meta/base_preds/ in .json files




#base_preds/global_small_valid_large_test_size_-0.1_v2_Hist_k5_0.04.parquet