from model_utils import main, run_category_based_training
import argparse

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Run NRS evaluation for specified cluster(s).")
     parser.add_argument("--dataset", type=str, default="train", help="Dataset type: train or valid")
     parser.add_argument("--cluster_id", type=str, default=None,
                         help="Cluster ID (or comma-separated list) to evaluate (if not provided, all clusters will be processed)")
     parser.add_argument("--dataset_size", type=str, default="large", help="Dataset size: large or small")
     parser.add_argument("--process_dfs", action='store_true', help="Process dataframes if needed")
     parser.add_argument("--process_behaviors", action='store_true', help="Process behaviors if needed")
     parser.add_argument("--train_all", action='store_true', help="train all models even if old exists")
     args = parser.parse_args()
     train_only_new = not args.train_all
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
#run_category_based_training(dataset_size, data_dir_train, valid_data_dir, zip_file, valid_zip_file, news_file='news.tsv', behaviors_file='behaviors.tsv'):
     run_category_based_training(dataset_size=args.dataset_size,
         data_dir_train=data_dir_train, valid_data_dir=data_dir_valid,
         zip_file=zip_file_train, valid_zip_file=zip_file_valid,
         news_file='news.tsv',
         behaviors_file='behaviors.tsv', train_only_new=train_only_new)

