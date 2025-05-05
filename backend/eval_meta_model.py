from model_utils import evaluate_meta_from_parquet
import argparse, pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NRS evaluation for specified cluster(s).")
    parser.add_argument("--dataset", type=str, default="train", help="Dataset type: train or valid")
    parser.add_argument("--train_dataset", type=str, default="train", help="Dataset type meta model was trained on: train or valid")
    parser.add_argument("--model_type", type=str, default="cluster", help="model type: cluster or category")
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
    parser.add_argument("--test_size", type=float, default=-0.1, help="test_size which validation scores meta model is trained on. Ignored when given negative value")


    args = parser.parse_args()

    meta_name = args.meta_model_type
    if args.tree_method != '':
        meta_name = f"{meta_name}_{args.tree_method}"
    if args.booster != 'default':
        meta_name = f"{meta_name}_{args.booster}"
    #pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}_test_size_{args.test_size}.parquet"
    pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}.parquet"
    meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{args.model_type}_{args.model_size}_{args.train_dataset}_{args.train_dataset_size}_test_size_{args.test_size}"
    if args.test_size < 0:
        meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{args.model_type}_{args.model_size}_{args.train_dataset}_{args.train_dataset_size}"
        pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}.parquet"
    meta_path = f"meta/{meta_model_base}.joblib"
    store_metrics_path = f"meta/results/{meta_model_base}_{args.dataset}.json"
    store_parquet_path = f"meta/{meta_model_base}_{args.dataset}.parquet"
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

