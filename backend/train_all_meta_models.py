from model_utils import main, train_meta_from_parquet
import argparse, pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NRS evaluation for specified cluster(s).")
    parser.add_argument("--dataset", type=str, default="valid", help="Dataset type: train or valid")
    parser.add_argument("--model_type", type=str, default="every", help="model type: cluster or category")
    parser.add_argument("--load_best_model", action='store_true', help="Load best model")
    parser.add_argument("--cluster_id", type=str, default=None,
                        help="Cluster ID (or comma-separated list) to evaluate (if not provided, all clusters will be processed)")
    parser.add_argument("--dataset_size", type=str, default="small", help="Dataset size: large or small")
    parser.add_argument("--model_size", type=str, default="small", help="Model size: large or small")
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
    parser.add_argument("--test_size", type=float, default=-0.1, help="test_size which validation scores meta model is trained on. Ignored when given negative value")

    parser.add_argument("--eval_frac", type=float, default=1.0, help="eval_frac")

    args = parser.parse_args()
    if args.dataset_size == "small":
        data_dir_train="dataset/small/train/"
        data_dir_valid="dataset/small/valid/"
        zip_file_train="MINDsmall_train.zip"
        zip_file_valid="MINDsmall_dev.zip"
        user_category_profiles_path="small_user_category_profiles.pkl"
        user_cluster_df_path="small_user_cluster_df.pkl"
        cluster_id=args.cluster_id
    else:
        data_dir_train="dataset/train/"
        data_dir_valid="dataset/valid/"
        zip_file_train="MINDlarge_train.zip"
        zip_file_valid="MINDlarge_dev.zip"
        user_category_profiles_path="user_category_profiles.pkl"
        user_cluster_df_path="user_cluster_df.pkl"
        cluster_id=args.cluster_id
    if args.retrain != "none":
        retrain_models = args.retrain.split(',')
    else:
        retrain_models = []
    meta_name = args.meta_model_type
    if args.tree_method != '':
        meta_name = f"{meta_name}_{args.tree_method}"
    if args.booster != 'default':
        meta_name = f"{meta_name}_{args.booster}"
    vers_suffix = ""
    if args.v > 1:
        vers_suffix = f"_v{args.v}"
    dataset_type_suffix = ""
    if args.col != "":
        dataset_type_suffix = f"_{args.col}"
    eval_frac_suffix = ""
    if args.eval_frac < 1.0:
        eval_frac_suffix = f"_{args.eval_frac}"

    if args.model_type == "every":
        model_type = "all"
    else:
        model_type = args.model_type
    pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}_test_size_{args.test_size}{vers_suffix}{dataset_type_suffix}{eval_frac_suffix}.parquet"
    meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{model_type}_{args.model_size}_{args.dataset}_{args.dataset_size}_test_size_{args.test_size}{vers_suffix}"
    out_path = f"meta/{meta_model_base}.joblib"
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_meta_from_parquet(
        parquet_dir=args.parquet_dir,
        pattern=pattern,
        out_path=out_path,
        estimator=args.estimator,
        base_model_type=model_type,
        meta_model_type=args.meta_model_type,
        booster=args.booster,
        tree_method = args.tree_method,
        n_estimators = args.n_estimators)

    if args.model_type == "every":
        model_type = "category"
    else:
        model_type = args.model_type
    pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}_test_size_{args.test_size}{vers_suffix}{dataset_type_suffix}{eval_frac_suffix}.parquet"
    meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{model_type}_{args.model_size}_{args.dataset}_{args.dataset_size}_test_size_{args.test_size}{vers_suffix}"
    out_path = f"meta/{meta_model_base}.joblib"
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_meta_from_parquet(
        parquet_dir=args.parquet_dir,
        pattern=pattern,
        out_path=out_path,
        estimator=args.estimator,
        base_model_type=model_type,
        meta_model_type=args.meta_model_type,
        booster=args.booster,
        tree_method = args.tree_method,
        n_estimators = args.n_estimators)


    if args.model_type == "every":
        model_type = "cluster"
    else:
        model_type = args.model_type
    pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}_test_size_{args.test_size}{vers_suffix}{dataset_type_suffix}{eval_frac_suffix}.parquet"
    meta_model_base = f"meta_model_{meta_name}_{args.n_estimators}_{model_type}_{args.model_size}_{args.dataset}_{args.dataset_size}_test_size_{args.test_size}{vers_suffix}"
    out_path = f"meta/{meta_model_base}.joblib"
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_meta_from_parquet(
        parquet_dir=args.parquet_dir,
        pattern=pattern,
        out_path=out_path,
        estimator=args.estimator,
        base_model_type=model_type,
        meta_model_type=args.meta_model_type,
        booster=args.booster,
        tree_method = args.tree_method,
        n_estimators = args.n_estimators)


# TRAIN DEFAULT MODELS WITH COMMAND: python3 train_all_models.py
# EVAL DEFAULT MODEL WITH COMMAND: python3 eval_all_models.py
# TRAIN DEFAULT META MODELS WITH COMMAND: python3 train_all_meta_models.py
# EVAL DEFAULT META MODELS WITH COMMAND: python3 eval_all_meta_models.py

# Meta models' results are found at backend/meta/results/ in .json files
# Base models' results are found at backend/meta/base_preds/ in .json files