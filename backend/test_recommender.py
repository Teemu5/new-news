from model_utils import main, log_print
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NRS evaluation for specified cluster(s).")
    parser.add_argument("--dataset", type=str, default="train", help="Dataset type: train or valid")
    parser.add_argument("--cluster_id", type=str, default=None,
                        help="Cluster ID (or comma-separated list) to evaluate (if not provided, all clusters will be processed)")
    parser.add_argument("--dataset_size", type=str, default="large", help="Dataset size: large or small")
    parser.add_argument("--valid_dataset_size", type=str, default="default", help="Validation Dataset size: large or small")
    parser.add_argument("--ext_dataset_size", type=str, default="large", help="Dataset size: large or small")
    parser.add_argument("--process_dfs", action='store_true', help="Process dataframes if needed")
    parser.add_argument("--process_behaviors", action='store_true', help="Process behaviors if needed")
    parser.add_argument("--dont_resume", action='store_true', help="add to start from beginning")
    parser.add_argument("--model_type", type=str, default="cluster", help="model_type: cluster or category")
    parser.add_argument("--model_size", type=str, default="large", help="model_size: large or small")
    parser.add_argument("--eval_scope", type=str, default="cluster", help="model_type: cluster or global")
    parser.add_argument("--epochs", type=int, default=1, help="epochs number of (global) model to train/load")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size for model training")
    parser.add_argument("--adaptivity_test", action='store_true', help="run adaptivity test")
    parser.add_argument("--shuffle", action='store_true', help="shuffle user list")
    parser.add_argument("--load_best_model", action='store_true', help="Load best model")
    parser.add_argument("--eval_separate", action='store_true', help="eval_separate models")
    parser.add_argument("--use_full_eval_separate_set", action='store_true', help="use_full_eval_separate_set models")
    parser.add_argument("--skip_already_evaluated", action='store_true', help="skip_already_evaluated skips evaluation if results exist")
    parser.add_argument("--load_best_models", type=str, default="", help="Load best models for these types in comma separated list")
    parser.add_argument("--retrain_models", type=str, default="", help="retrain_models for these types in comma separated list")
    parser.add_argument("--drift_fraction", type=float, default=0.5, help="Fraction of history to swap during adaptivity test (0.0–1.0)")
    parser.add_argument("--n_estimators", type=int, default=300, help="n_estimators for meta model name")
    parser.add_argument("--test_size", type=float, default=-0.1, help="test_size which validation scores meta model is trained on")
    parser.add_argument("--use_model_base", action='store_true', help="Set to model_base naming for models")
    args = parser.parse_args()
    log_print(f"test_recommender.py with args: {parser.parse_args()}")
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
    ext_data_dir_train="dataset/train/"
    ext_data_dir_valid="dataset/valid/"
    ext_zip_file_train="MINDlarge_train.zip"
    ext_zip_file_valid="MINDlarge_dev.zip"
    if args.ext_dataset_size == "small":
        ext_data_dir_train="dataset/small/train/"
        ext_data_dir_valid="dataset/small/valid/"
        ext_zip_file_train="MINDsmall_train.zip"
        ext_zip_file_valid="MINDsmall_dev.zip"
    vsize = args.valid_dataset_size
    if args.valid_dataset_size == "default":
        vsize = args.dataset_size
    main(dataset=args.dataset,
        process_dfs=args.process_dfs, process_behaviors=args.process_behaviors,
        data_dir_train=data_dir_train, data_dir_valid=data_dir_valid,
        zip_file_train=zip_file_train, zip_file_valid=zip_file_valid,
        user_category_profiles_path=user_category_profiles_path,
        user_cluster_df_path=user_cluster_df_path, cluster_id=args.cluster_id, resume=not args.dont_resume,
        model_type=args.model_type, dataset_size=args.dataset_size, load_best_model=args.load_best_model, load_best_models=args.load_best_models.split(','), eval_scope=args.eval_scope, model_size=args.model_size, epochs=args.epochs,
        adaptivity_test=args.adaptivity_test, shuffle=args.shuffle, drift_fraction=args.drift_fraction, eval_separate=args.eval_separate, use_full_eval_separate_set=args.use_full_eval_separate_set,
        skip_already_evaluated=args.skip_already_evaluated, batch_size=args.batch_size, retrain_models=args.retrain_models.split(','),
        ext_data_dir_train=ext_data_dir_train,ext_data_dir_valid=ext_data_dir_valid,
        ext_zip_file_train=ext_zip_file_train,ext_zip_file_valid=ext_zip_file_valid,n_estimators=args.n_estimators, test_size=args.test_size,
        use_model_base=args.use_model_base, valid_dataset_size=vsize)
