from model_utils import main, run_meta_training, train_meta_from_parquet
import argparse, pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NRS evaluation for specified cluster(s).")
    parser.add_argument("--dataset", type=str, default="train", help="Dataset type: train or valid")
    parser.add_argument("--model_type", type=str, default="cluster", help="model type: cluster or category")
    parser.add_argument("--load_best_model", action='store_true', help="Load best model")
    parser.add_argument("--cluster_id", type=str, default=None,
                        help="Cluster ID (or comma-separated list) to evaluate (if not provided, all clusters will be processed)")
    parser.add_argument("--dataset_size", type=str, default="large", help="Dataset size: large or small")
    parser.add_argument("--model_size", type=str, default="large", help="Model size: large or small")
    parser.add_argument("--dont_resume", action='store_true', help="dont resume from cp")
    parser.add_argument("--process_dfs", action='store_true', help="Process dataframes if needed")
    parser.add_argument("--process_behaviors", action='store_true', help="Process behaviors if needed")
    parser.add_argument("--retrain", type=str, default="none", help="comma separate list of model types to retrain")
    parser.add_argument("--epochs", type=int, default=1, help="epochs number of (global) model to train/load")
    parser.add_argument("--meta_model_type", type=str, default="LogisticRegressionCV", help="meta model type LogisticRegression, LogisticRegressionCV or SGDClassifier")
    parser.add_argument("--booster", type=str, default="default", help="booster for XGBClassifier")
    parser.add_argument("--tree_method", type=str, default="default", help="tree_method for XGBClassifier")


    parser.add_argument("--parquet_dir", type=str, default="base_preds", help="parquet_dir for base preds")
    parser.add_argument("--estimator", type=str, default="lbfgs", help="estimator for meta model")

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
    pattern = f"*_{args.model_size}_{args.dataset}_{args.dataset_size}.parquet"
    meta_name = args.meta_model_type
    if args.tree_method != 'default':
        meta_name = f"{meta_name}_{args.tree_method}"
    if args.booster != 'default':
        meta_name = f"{meta_name}_{args.booster}"
    meta_model_base = f"meta_model_{meta_name}_{args.model_type}_{args.model_size}_{args.dataset}_{args.dataset_size}"
    out_path = f"meta/{meta_model_base}.joblib"
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_meta_from_parquet(
        parquet_dir=args.parquet_dir,
        pattern=pattern,
        out_path=out_path,
        estimator=args.estimator,
        base_model_type=args.model_type,
        meta_model_type=args.meta_model_type,
        booster=args.booster,
        tree_method = args.tree_method)
"""
     run_meta_training(dataset=args.dataset, dataset_size=args.dataset_size, model_type=args.model_type,
         process_dfs=args.process_dfs, process_behaviors=args.process_behaviors,
         data_dir_train=data_dir_train, data_dir_valid=data_dir_valid,
         zip_file_train=zip_file_train, zip_file_valid=zip_file_valid,
         user_category_profiles_path=user_category_profiles_path, resume=not args.dont_resume,
         user_cluster_df_path=user_cluster_df_path, cluster_id=args.cluster_id, load_best_model=args.load_best_model, model_size=args.model_size, retrain_models=retrain_models, epochs=args.epochs)
"""