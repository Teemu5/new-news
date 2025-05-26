from model_utils import main, log_print
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NRS evaluation for specified cluster(s).")
    parser.add_argument("--dataset", type=str, default="train", help="Dataset type: train or valid")
    parser.add_argument("--cluster_id", type=str, default=None,
                        help="Cluster ID (or comma-separated list) to evaluate (if not provided, all clusters will be processed)")
    parser.add_argument("--dataset_size", type=str, default="small", help="Dataset size: large or small")
    parser.add_argument("--valid_dataset_size", type=str, default="default", help="Validation Dataset size: large or small")
    parser.add_argument("--ext_dataset_size", type=str, default="large", help="Dataset size: large or small")
    parser.add_argument("--process_dfs", action='store_true', help="Process dataframes if needed")
    parser.add_argument("--process_behaviors", action='store_true', help="Process behaviors if needed")
    parser.add_argument("--dont_resume", action='store_true', help="add to start from beginning")
    parser.add_argument("--model_type", type=str, default="all", help="model_type: cluster or category")
    parser.add_argument("--model_size", type=str, default="small", help="model_size: large or small")
    parser.add_argument("--eval_scope", type=str, default="cluster", help="model_type: cluster or global")
    parser.add_argument("--epochs", type=int, default=1, help="epochs number of (global) model to train/load")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size for model training")
    parser.add_argument("--adaptivity_test", action='store_true', help="run adaptivity test")
    parser.add_argument("--shuffle", action='store_true', help="shuffle user list")
    parser.add_argument("--load_best_model", action='store_true', help="Load best model")
    parser.add_argument("--eval_separate", action='store_true', help="eval_separate models")
    parser.add_argument("--eval", action='store_true', help="eval_separate models")
    parser.add_argument("--evaluate", action='store_true', help="eval_separate models")
    parser.add_argument("--use_full_eval_separate_set", action='store_true', help="use_full_eval_separate_set models")
    parser.add_argument("--skip_already_evaluated", action='store_true', help="skip_already_evaluated skips evaluation if results exist")
    parser.add_argument("--load_best_models", type=str, default="", help="Load best models for these types in comma separated list")
    parser.add_argument("--retrain_models", type=str, default="", help="retrain_models for these types in comma separated list")
    parser.add_argument("--drift_fraction", type=float, default=0.5, help="Fraction of history to swap during adaptivity test (0.0–1.0)")
    parser.add_argument("--n_estimators", type=int, default=300, help="n_estimators for meta model name")
    parser.add_argument("--test_size", type=float, default=-0.1, help="test_size which validation scores meta model is trained on")
    parser.add_argument("--use_model_base", action='store_true', help="Set to model_base naming for models")
    parser.add_argument("--meta_train_dataset", type=str, default="valid", help="Meta model dataset type: train or valid")
    parser.add_argument("--dont_pad_zeros", action='store_true', help="Set to not pad user history with zeros in experiments. This should mean that padding with avg profile if not padded with zeros")
    parser.add_argument("--use_full_avg_profile", action='store_true', help="Set to use full avg profile in place of actual user history for all users.")
    parser.add_argument("--process_valid_sets", action='store_true', help="process_valid_sets")
    parser.add_argument("--end_after_process", action='store_true', help="end_after_process, ends after preprocessing")
    parser.add_argument("--make_title_tensor", action='store_true', help="make_title_tensor, forces making new title tensor")
    parser.add_argument("--eval_frac", type=float, default=1.0, help="eval_fracion used")
    parser.add_argument("--big_tokenizer", action='store_true', help="big_tokenizer, use big tokenizer combined from all sets (train set + both validation sets)")
    parser.add_argument("--pivots", type=str, default="", help="pivots in comma separated list")
    parser.add_argument("--ks", type=str, default="", help="ks in comma separated list")
    parser.add_argument("--wait_at_start", type=int, default=0, help="wait for x minutes")
    parser.add_argument("--reverse", action='store_true', help="reverse, eval order")
    parser.add_argument("--tune_threshold", action='store_true', help="tune_threshold")
    parser.add_argument("--regenerate_metrics", action='store_true', help="regenerate_metrics")
    parser.add_argument("--v", type=int, default=4, help="version")
    parser.add_argument("--model_arc_type", type=str, default="fastformer", help="model_arc_type")
    parser.add_argument("--timed", action='store_true', help="timed")
    parser.add_argument("--use_cpu", action='store_true', help="use_cpu")
    parser.add_argument("--clear_store", action='store_true', help="clear_store")
    parser.add_argument("--dont_process_all", action='store_true', help="dont preprocess")
    args = parser.parse_args()
    if args.dont_process_all:
        process_valid_sets=args.process_valid_sets
        process_behaviors=args.process_behaviors
        process_dfs=args.process_dfs
    else:
        process_valid_sets=True
        process_behaviors=True
        process_dfs=True
    print(f"wait for {args.wait_at_start}mins")
    time.sleep(args.wait_at_start * 60)
    log_print(f"test_recommender.py with args: {parser.parse_args()}")
    vers_suffix = ""
    if args.v > 1:
        vers_suffix = f"_v{args.v}"
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
    pivots = [int(n) for n in args.pivots.split(',')] if args.pivots != "" else []
    ks = [int(n) for n in args.ks.split(',')] if args.ks != "" else []
    evaluate = args.eval_separate or args.eval or args.evaluate
    main(dataset=args.dataset,
        process_dfs=process_dfs, process_behaviors=process_behaviors,
        data_dir_train=data_dir_train, data_dir_valid=data_dir_valid,
        zip_file_train=zip_file_train, zip_file_valid=zip_file_valid,
        user_category_profiles_path=user_category_profiles_path,
        user_cluster_df_path=user_cluster_df_path, cluster_id=args.cluster_id, resume=not args.dont_resume,
        model_type=args.model_type, dataset_size=args.dataset_size, load_best_model=args.load_best_model, load_best_models=args.load_best_models.split(','), eval_scope=args.eval_scope, model_size=args.model_size, epochs=args.epochs,
        adaptivity_test=args.adaptivity_test, shuffle=args.shuffle, drift_fraction=args.drift_fraction, eval_separate=evaluate, use_full_eval_separate_set=args.use_full_eval_separate_set,
        skip_already_evaluated=args.skip_already_evaluated, batch_size=args.batch_size, retrain_models=args.retrain_models.split(','),
        ext_data_dir_train=ext_data_dir_train,ext_data_dir_valid=ext_data_dir_valid,
        ext_zip_file_train=ext_zip_file_train,ext_zip_file_valid=ext_zip_file_valid,n_estimators=args.n_estimators, test_size=args.test_size,
        use_model_base=args.use_model_base, valid_dataset_size=vsize, meta_train_dataset=args.meta_train_dataset, pad_zeros=not args.dont_pad_zeros,
        use_full_avg_profile=args.use_full_avg_profile, process_valid_sets=process_valid_sets, end_after_process=args.end_after_process,
        make_title_tensor=args.make_title_tensor, eval_frac=args.eval_frac, big_tokenizer=args.big_tokenizer,
        pivots = pivots, ks = ks, reverse=args.reverse, tune_threshold=args.tune_threshold, regenerate_metrics=args.regenerate_metrics, vers_suffix=vers_suffix,
        model_arc_type=args.model_arc_type, timed=args.timed, use_cpu=args.use_cpu, clear_store=args.clear_store)


    model_type = "global"
    model_arc_type = "nrms"
    main(dataset=args.dataset,
        process_dfs=process_dfs, process_behaviors=process_behaviors,
        data_dir_train=data_dir_train, data_dir_valid=data_dir_valid,
        zip_file_train=zip_file_train, zip_file_valid=zip_file_valid,
        user_category_profiles_path=user_category_profiles_path,
        user_cluster_df_path=user_cluster_df_path, cluster_id=args.cluster_id, resume=not args.dont_resume,
        model_type=model_type, dataset_size=args.dataset_size, load_best_model=args.load_best_model, load_best_models=args.load_best_models.split(','), eval_scope=args.eval_scope, model_size=args.model_size, epochs=args.epochs,
        adaptivity_test=args.adaptivity_test, shuffle=args.shuffle, drift_fraction=args.drift_fraction, eval_separate=evaluate, use_full_eval_separate_set=args.use_full_eval_separate_set,
        skip_already_evaluated=args.skip_already_evaluated, batch_size=args.batch_size, retrain_models=args.retrain_models.split(','),
        ext_data_dir_train=ext_data_dir_train,ext_data_dir_valid=ext_data_dir_valid,
        ext_zip_file_train=ext_zip_file_train,ext_zip_file_valid=ext_zip_file_valid,n_estimators=args.n_estimators, test_size=args.test_size,
        use_model_base=args.use_model_base, valid_dataset_size=vsize, meta_train_dataset=args.meta_train_dataset, pad_zeros=not args.dont_pad_zeros,
        use_full_avg_profile=args.use_full_avg_profile, process_valid_sets=process_valid_sets, end_after_process=args.end_after_process,
        make_title_tensor=args.make_title_tensor, eval_frac=args.eval_frac, big_tokenizer=args.big_tokenizer,
        pivots = pivots, ks = ks, reverse=args.reverse, tune_threshold=args.tune_threshold, regenerate_metrics=args.regenerate_metrics, vers_suffix=vers_suffix,
        model_arc_type=model_arc_type, timed=args.timed, use_cpu=args.use_cpu, clear_store=args.clear_store)

    # Train timed models next!
    timed = True
    main(dataset=args.dataset,
        process_dfs=process_dfs, process_behaviors=process_behaviors,
        data_dir_train=data_dir_train, data_dir_valid=data_dir_valid,
        zip_file_train=zip_file_train, zip_file_valid=zip_file_valid,
        user_category_profiles_path=user_category_profiles_path,
        user_cluster_df_path=user_cluster_df_path, cluster_id=args.cluster_id, resume=not args.dont_resume,
        model_type=model_type, dataset_size=args.dataset_size, load_best_model=args.load_best_model, load_best_models=args.load_best_models.split(','), eval_scope=args.eval_scope, model_size=args.model_size, epochs=args.epochs,
        adaptivity_test=args.adaptivity_test, shuffle=args.shuffle, drift_fraction=args.drift_fraction, eval_separate=evaluate, use_full_eval_separate_set=args.use_full_eval_separate_set,
        skip_already_evaluated=args.skip_already_evaluated, batch_size=args.batch_size, retrain_models=args.retrain_models.split(','),
        ext_data_dir_train=ext_data_dir_train,ext_data_dir_valid=ext_data_dir_valid,
        ext_zip_file_train=ext_zip_file_train,ext_zip_file_valid=ext_zip_file_valid,n_estimators=args.n_estimators, test_size=args.test_size,
        use_model_base=args.use_model_base, valid_dataset_size=vsize, meta_train_dataset=args.meta_train_dataset, pad_zeros=not args.dont_pad_zeros,
        use_full_avg_profile=args.use_full_avg_profile, process_valid_sets=process_valid_sets, end_after_process=args.end_after_process,
        make_title_tensor=args.make_title_tensor, eval_frac=args.eval_frac, big_tokenizer=args.big_tokenizer,
        pivots = pivots, ks = ks, reverse=args.reverse, tune_threshold=args.tune_threshold, regenerate_metrics=args.regenerate_metrics, vers_suffix=vers_suffix,
        model_arc_type=model_arc_type, timed=timed, use_cpu=args.use_cpu, clear_store=args.clear_store)

    model_arc_type = "fastformer"
    main(dataset=args.dataset,
        process_dfs=process_dfs, process_behaviors=process_behaviors,
        data_dir_train=data_dir_train, data_dir_valid=data_dir_valid,
        zip_file_train=zip_file_train, zip_file_valid=zip_file_valid,
        user_category_profiles_path=user_category_profiles_path,
        user_cluster_df_path=user_cluster_df_path, cluster_id=args.cluster_id, resume=not args.dont_resume,
        model_type=model_type, dataset_size=args.dataset_size, load_best_model=args.load_best_model, load_best_models=args.load_best_models.split(','), eval_scope=args.eval_scope, model_size=args.model_size, epochs=args.epochs,
        adaptivity_test=args.adaptivity_test, shuffle=args.shuffle, drift_fraction=args.drift_fraction, eval_separate=evaluate, use_full_eval_separate_set=args.use_full_eval_separate_set,
        skip_already_evaluated=args.skip_already_evaluated, batch_size=args.batch_size, retrain_models=args.retrain_models.split(','),
        ext_data_dir_train=ext_data_dir_train,ext_data_dir_valid=ext_data_dir_valid,
        ext_zip_file_train=ext_zip_file_train,ext_zip_file_valid=ext_zip_file_valid,n_estimators=args.n_estimators, test_size=args.test_size,
        use_model_base=args.use_model_base, valid_dataset_size=vsize, meta_train_dataset=args.meta_train_dataset, pad_zeros=not args.dont_pad_zeros,
        use_full_avg_profile=args.use_full_avg_profile, process_valid_sets=process_valid_sets, end_after_process=args.end_after_process,
        make_title_tensor=args.make_title_tensor, eval_frac=args.eval_frac, big_tokenizer=args.big_tokenizer,
        pivots = pivots, ks = ks, reverse=args.reverse, tune_threshold=args.tune_threshold, regenerate_metrics=args.regenerate_metrics, vers_suffix=vers_suffix,
        model_arc_type=model_arc_type, timed=timed, use_cpu=args.use_cpu, clear_store=args.clear_store)
    
# TRAIN DEFAULT MODELS WITH COMMAND: python3 train_all_models.py
# EVAL DEFAULT MODEL WITH COMMAND: python3 eval_all_models.py
# TRAIN DEFAULT META MODELS WITH COMMAND: python3 train_all_meta_models.py
# EVAL DEFAULT META MODELS WITH COMMAND: python3 eval_all_meta_models.py

# Meta models' results are found at backend/meta/results/ in .json files
# Base models' results are found at backend/meta/base_preds/ in .json files