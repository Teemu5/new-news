
import datetime
from datetime import timedelta
import glob
import hashlib
import logging
import math
import os
import pickle
import re
import shutil
import time
import zipfile
from traceback import format_exc

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from dateutil.parser import isoparse

from sklearn.metrics import precision_recall_curve
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    silhouette_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_fscore_support,
    average_precision_score,
    ndcg_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from joblib import Parallel, delayed
import joblib

import keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Dot,
    Dropout,
    Embedding,
    Input,
    Layer,
    LayerNormalization,
    Multiply,
    Softmax,
    TimeDistributed,
    MultiHeadAttention,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, register_keras_serializable, custom_object_scope
from keras.models import Model, model_from_json

from huggingface_hub import hf_hub_download
from pathlib import Path

from recommender import (
    fastformer_model_predict,
    ensemble_bagging,
    ensemble_boosting,
    train_stacking_meta_model,
    ensemble_stacking,
    hybrid_ensemble,
)
from joblib import Memory
import fnmatch
memory = Memory("./user_score_cache")
GLOBAL_MODEL_DICT: dict[str, tf.keras.Model] = {}

def set_global_models(md: dict[str, tf.keras.Model]):
    global GLOBAL_MODEL_DICT
    GLOBAL_MODEL_DICT = md

@memory.cache
def cached_score_single(history_arr: np.ndarray,
                        candidates_arr: np.ndarray,
                        model_key: str,
                        batch_size: int = 128) -> np.ndarray:
    if history_arr.ndim == 3 and history_arr.shape[0] == 1:
        history_arr = np.squeeze(history_arr, axis=0)        # (50, 30)

    if candidates_arr.ndim == 3 and candidates_arr.shape[1] == 1:
        candidates_arr = np.squeeze(candidates_arr, axis=1)  # (N, 30)

    hist_tensor = tf.convert_to_tensor(history_arr, dtype=tf.int32)  # (50, 30)
    cand_tensor = tf.convert_to_tensor(candidates_arr, dtype=tf.int32)  # (N, 30)

    N = cand_tensor.shape[0]
    hist_batch = tf.repeat(hist_tensor[None, ...], repeats=N, axis=0)  # (N, 50, 30)

    model = GLOBAL_MODEL_DICT[model_key]
    preds = model.predict(
        {"history_input": hist_batch,
         "candidate_input": cand_tensor},
        batch_size=batch_size
    )
    return preds.ravel()

    #return score_candidates_in_batch(history_tensor, candidate_tensors, model, batch_size)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))
def clean_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if not w in stop_words]
    return ' '.join(words)

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/cluster_profile_{date_str}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_checkpoint_directory(index, base_dir="checkpoints"):
    part = index // 100000 + 1
    sub = (index % 100000) // 10000 + 1
    return os.path.join(base_dir, f"part{part}", f"{sub}")


def save_checkpoint(checkpoint_index, delta_samples, base_dir="checkpoints"):
    dir_name = get_checkpoint_directory(checkpoint_index, base_dir)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, f"checkpoint_{checkpoint_index}.pkl")
    checkpoint_data = {"delta_samples": delta_samples, "last_index": checkpoint_index}
    with open(filename, "wb") as f:
        pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Checkpoint saved at row {checkpoint_index+1} in {filename}")

def load_checkpoints(base_dir="checkpoints"):
    pattern = os.path.join(base_dir, "part*", "*", "checkpoint_*.pkl")
    cp_files = glob.glob(pattern)

    cp_files = sorted(cp_files, key=lambda f: int(os.path.basename(f).replace("checkpoint_", "").replace(".pkl", "")))
    
    merged_samples = []
    max_index = -1
    for cp_file in cp_files:
        print(f"Loading checkpoint from {cp_file}")
        with open(cp_file, "rb") as f:
            data = pickle.load(f)
        merged_samples.extend(data["delta_samples"])
        max_index = max(max_index, data["last_index"])
    start_index = max_index + 1 if max_index >= 0 else 0
    return merged_samples, start_index

def prepare_category_train_dfs(data_dir, news_file, behaviors_file,
                               max_title_length=30, max_history_length=50,
                               save_filename='category_train_dfs.pkl'):
    if os.path.exists(save_filename):
        print(f"Loading precomputed category training data from {save_filename}")
        with open(save_filename, "rb") as f:
            category_train_dfs, news_df, behaviors_df, tokenizer = pickle.load(f)
        print("Loaded category training data.")
        return category_train_dfs, news_df, behaviors_df, tokenizer

    print("Precomputed category training data not found. Computing now...")

    # Load news data
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    print(f"Loaded news data from {news_path}: {news_df.shape}")

    # Clean titles and abstracts and create combined text
    news_df['CleanTitle'] = news_df['Title'].apply(clean_text)
    news_df['CleanAbstract'] = news_df['Abstract'].apply(clean_text)
    news_df['CombinedText'] = news_df['CleanTitle'] + " " + news_df['CleanAbstract']
    news_df["CombinedText"] = news_df["CombinedText"].astype(str).fillna("")

    # Load behaviors data
    behaviors_path = os.path.join(data_dir, behaviors_file)
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna("")
    print(f"Loaded behaviors data from {behaviors_path}: {behaviors_df.shape}")

    # Fit the tokenizer on news CombinedText
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())

    # Create mapping from NewsID to padded text
    news_df['EncodedText'] = tokenizer.texts_to_sequences(news_df['CombinedText'])
    news_df['PaddedText'] = list(pad_sequences(news_df['EncodedText'],
                                                maxlen=max_title_length, padding='post', truncating='post'))
    news_text_dict = dict(zip(news_df['NewsID'], news_df['PaddedText']))

    def parse_impressions(impressions):
        impression_list = impressions.split()
        news_ids = []
        labels = []
        for imp in impression_list:
            try:
                news_id, label = imp.split('-')
                news_ids.append(news_id)
                labels.append(int(label))
            except ValueError:
                continue
        return news_ids, labels

    behaviors_df[['ImpressionNewsIDs', 'ImpressionLabels']] = behaviors_df['Impressions'].apply(
        lambda x: pd.Series(parse_impressions(x))
    )

    samples = []
    checkpoint_base_dir = "checkpoints"
    checkpoint_interval = 1000  # Save a checkpoint every 1000 rows.
    start_index, prev_checkpoint_index = 0, 0

    # Try loading previously saved checkpoints from the hierarchical structure.
    loaded_samples, loaded_index = load_checkpoints(checkpoint_base_dir)
    if loaded_index > 0:
        samples = loaded_samples
        start_index = loaded_index
        prev_checkpoint_index = loaded_index
        print(f"Loaded checkpoint parts. Resuming from row index {start_index}")
    else:
        print("No valid checkpoint parts found. Starting from the beginning.")

    print("Building training samples...")
    total_rows = behaviors_df.shape[0]
    for i, row in tqdm(behaviors_df.iterrows(), total=total_rows, desc="Building samples"):
        if i < start_index:
            continue
        user_id = row['UserID']
        # Process user history: get candidate training sample history
        history_ids = row['HistoryText'].split() if row['HistoryText'] != "" else []
        history_texts = [news_text_dict.get(nid, [0] * max_title_length) for nid in history_ids]
        if len(history_texts) < max_history_length:
            padding = [[0] * max_title_length] * (max_history_length - len(history_texts))
            history_texts = padding + history_texts
        else:
            history_texts = history_texts[-max_history_length:]

        candidate_news_ids = row['ImpressionNewsIDs']
        labels = row['ImpressionLabels']
        for candidate_news_id, label in zip(candidate_news_ids, labels):
            candidate_text = news_text_dict.get(candidate_news_id, [0] * max_title_length)
            candidate_category_series = news_df[news_df['NewsID'] == candidate_news_id]['Category']
            candidate_category = candidate_category_series.iloc[0] if not candidate_category_series.empty else "Unknown"
            samples.append({
                'UserID': user_id,
                'HistoryTitles': history_texts,
                'CandidateTitleTokens': candidate_text,
                'Label': label,
                'CandidateCategory': candidate_category
            })
        # Save a checkpoint when reaching the interval.
        if (i + 1) % checkpoint_interval == 0:
            print(f"{i+1}/{total_rows} rows processed. Saving checkpoint...")
            # Only save the delta samples added since the last checkpoint.
            delta_samples = samples[prev_checkpoint_index: i+1]
            save_checkpoint(i, delta_samples, base_dir=checkpoint_base_dir)
            prev_checkpoint_index = i+1

    train_df = pd.DataFrame(samples)
    print(f"Created training DataFrame with {len(train_df)} samples.")

    # Group training samples by candidate category.
    category_train_dfs = dict(tuple(train_df.groupby('CandidateCategory')))
    for category, df in category_train_dfs.items():
        print(f"Category '{category}': {len(df)} samples.")

    # Save the complete computed data to disk.
    with open(save_filename, "wb") as f:
        pickle.dump((category_train_dfs, news_df, behaviors_df, tokenizer), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved category training data to {save_filename}")

    return category_train_dfs, news_df, behaviors_df, tokenizer


def run_category_based_training(dataset_size, data_dir_train, valid_data_dir, zip_file, valid_zip_file, news_file='news.tsv', behaviors_file='behaviors.tsv', train_only_new=True):
    print(f"unzipping datasets: data_dir_train={data_dir_train}, valid_data_dir={valid_data_dir}, zip_file={zip_file}, valid_zip_file={valid_zip_file}")
    unzip_datasets(data_dir_train, valid_data_dir, zip_file, valid_zip_file)
    print("Starting category-based training...")
    # Step 1: Load data and prepare training samples grouped by candidate category.
    category_train_dfs, news_df, behaviors_df, tokenizer = prepare_category_train_dfs(data_dir_train, news_file, behaviors_file, 30, 50, f"category_train_dfs_{dataset_size}.pkl")
    
    # Compute vocabulary size from the tokenizer.
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    max_history_length = 50
    max_title_length = 30
    
    # Step 2: Train a model for each category.
    category_models = train_category_models(category_train_dfs, vocab_size, max_history_length, max_title_length, 8, 3, dataset_size, train_only_new=train_only_new)
    
    print("Category-based training complete.")
    return category_models, news_df, behaviors_df, tokenizer


def get_candidate_category(candidate_id, news_df):
    candidate_category_series = news_df.loc[news_df['NewsID'] == candidate_id, 'Category']
    if not candidate_category_series.empty:
        return candidate_category_series.iloc[0]
    return "Unknown"

def build_meta_training_data_with_category(user_ids, behaviors_df, news_df, models_dict, tokenizer, 
                                           tfidf_vectorizer, cutoff_time, GLOBAL_AVG_PROFILE=None):
    # Build the meta-training data for a batch of user IDs, from base model predictions and candidate category information.

    logging.info("Starting batch_predict_users")
    predictions_list, candidate_ids_list, user_id_order = batch_predict_users(
        user_ids, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time, batch_size=128, GLOBAL_AVG_PROFILE=GLOBAL_AVG_PROFILE
    )
    logging.info("Finished batch_predict_users")
    
    X_meta_list = []
    y_meta_list = []
    candidate_categories_list = []  # To store the category for each candidate
    
    total_users = len(user_id_order)
    for idx, user_id in enumerate(user_id_order):
        candidate_ids = candidate_ids_list[idx]
        logging.info(f"Processing user {idx+1}/{total_users} in user_id_order")
        user_preds = predictions_list[idx]
        
        # Combine predictions from each model into a feature vector per candidate.
        features = np.column_stack([user_preds[key] for key in models_dict.keys()])
        
        # Get ground truth for this user.
        ground_truth = get_user_future_clicks(user_id, behaviors_df, cutoff_time)
        labels = np.array([1 if art in ground_truth else 0 for art in candidate_ids])
        
        X_meta_list.append(features)
        y_meta_list.append(labels)
        
        # Lookup and store candidate categories.
        for cand_id in candidate_ids:
            candidate_cat = get_candidate_category(cand_id, news_df)
            candidate_categories_list.append(candidate_cat)
        
        print(f"Processed meta data for user {idx+1}/{total_users}")
    
    # Stack features and labels from all users.
    if X_meta_list:
        X_meta_base = np.vstack(X_meta_list)  # Shape: (total_candidates, n_models)
        y_meta = np.hstack(y_meta_list)
    else:
        X_meta_base = np.empty((0, len(models_dict)))
        y_meta = np.empty(0)
        candidate_categories_list = np.empty((0,))
    
    # One-hot encode candidate categories.
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    candidate_categories_array = np.array(candidate_categories_list).reshape(-1, 1)
    cat_features = encoder.fit_transform(candidate_categories_array)
    
    # Concatenate the base prediction features and the candidate category features.
    X_meta = np.hstack([X_meta_base, cat_features])
    
    return X_meta, y_meta, encoder

def get_cache_filename(cluster_users, cutoff_time, cache_dir, base_name="ground_truth_freq", cluster_id=""):
    os.makedirs(cache_dir, exist_ok=True)
    # Create a unique string based on the sorted cluster users.
    users_str = ",".join(sorted(cluster_users))
    users_hash = hashlib.md5(users_str.encode('utf-8')).hexdigest()
    cutoff_time_str = pd.to_datetime(cutoff_time).strftime("%Y%m%d_%H%M%S")
    cache_filename = os.path.join(cache_dir, f"{base_name}_{cluster_id}_{cutoff_time_str}_{users_hash}.pkl")
    return cache_filename

def load_cache(cache_filename, force_new=False):
    if os.path.exists(cache_filename) and not force_new:
        logging.info(f"Loading cached data from {cache_filename}")
        with open(cache_filename, 'rb') as f:
            return pickle.load(f)
    return None
def get_cluster_ground_truth_frequency(cluster_users, behaviors_df, cutoff_time, cache_dir="cache", cluster_id = "", force_new = False):
    cache_filename = get_cache_filename(cluster_users, cutoff_time, cache_dir, "ground_truth_freq", cluster_id)
    cached_result = load_cache(cache_filename, force_new)
    if cached_result is not None:
        return cached_result

    # Ensure the Time column is in datetime format.
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    cutoff_time_dt = pd.to_datetime(cutoff_time)
    ground_truth_freq = {}
    total_users = len(cluster_users)
    for i, user_id in enumerate(cluster_users):
        user_future = behaviors_df[
            (behaviors_df['UserID'] == user_id) & (behaviors_df['Time'] > np.datetime64(cutoff_time_dt))
        ]
        for _, row in user_future.iterrows():
            if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
                continue
            for imp in row["Impressions"].split():
                try:
                    art, label = imp.split('-')
                    if int(label) == 1:
                        ground_truth_freq[art] = ground_truth_freq.get(art, 0) + 1
                except Exception as e:
                    print(f"Error parsing impression {imp}: {e}")
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1} of {total_users} users in get_cluster_ground_truth_frequency")
    # Save the result in the cache.
    with open(cache_filename, 'wb') as f:
        pickle.dump(ground_truth_freq, f)
    logging.info(f"Cached ground truth frequency to {cache_filename}")
    return ground_truth_freq

def compute_precision_recall_at_k(recommended_ids, ground_truth_ids, k, candidate_ids=None):
    if candidate_ids is None:
        candidate_ids = list(set(recommended_ids).union(ground_truth_ids))
    filtered_ground_truth = set(ground_truth_ids).intersection(set(candidate_ids))
    recommended_k = recommended_ids[:k]
    if not filtered_ground_truth:
        return 0.0, 0.0
    relevant = [1 if rec in filtered_ground_truth else 0 for rec in recommended_k]
    precision = sum(relevant) / k
    recall = sum(relevant) / len(filtered_ground_truth)
    return precision, recall

def get_cluster_ground_truth(cluster_users, behaviors_df, cutoff_time, cache_dir="cache", cluster_id = "", force_new = False):
    cache_filename = get_cache_filename(cluster_users, cutoff_time, cache_dir, "ground_truth", cluster_id)
    cached_result = load_cache(cache_filename, force_new)
    if cached_result is not None:
        return cached_result
    
    ground_truth = set()
    cutoff_time = pd.to_datetime(cutoff_time)
    total_users = len(cluster_users)
    for i, user_id in enumerate(cluster_users):
        user_future = behaviors_df[(behaviors_df['UserID'] == user_id) & (behaviors_df['Time'] > np.datetime64(cutoff_time))]
        for idx, row in user_future.iterrows():
            if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
                continue
            for imp in row["Impressions"].split():
                try:
                    art, label = imp.split('-')
                    if int(label) == 1:
                        ground_truth.add(art)
                except Exception as e:
                    print(f"Error parsing impression {imp}: {e}")
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1} of {total_users} users in get_cluster_ground_truth")
    # Save the result in the cache.
    with open(cache_filename, 'wb') as f:
        pickle.dump(ground_truth, f)
    logging.info(f"Cached ground truth frequency to {cache_filename}")
    return ground_truth

def tfidf_filter_candidates(candidates_df: pd.DataFrame, user_history_text: str, tfidf_vectorizer: TfidfVectorizer, min_similarity: float = 0.1, plot = False) -> pd.DataFrame:
    # filter based on tfidf min_similarity
    candidate_texts = candidates_df["CombinedText"].tolist() 
    candidate_vectors = tfidf_vectorizer.transform(candidate_texts)
    
    # Compute TF-IDF vector for the user's history
    user_vector = tfidf_vectorizer.transform([user_history_text])
    
    # Compute cosine similarities between user history and each candidate
    similarities = cosine_similarity(user_vector, candidate_vectors)[0]
    if plot:
        # Log the similarity distribution (for threshold determination)
        plt.hist(similarities, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel("TF-IDF Cosine Similarity")
        plt.ylabel("Frequency")
        plt.title("Distribution of Candidate Similarity Scores")
        plt.savefig("tfidf_similarity_distribution.png")
        plt.close()
        print("TF-IDF similarity distribution plotted and saved as 'tfidf_similarity_distribution.png'.")

    # Add the similarity score to the DataFrame
    candidates_df = candidates_df.copy()
    candidates_df["TFIDF_Similarity"] = similarities
    
    # Optionally filter out candidates below the threshold
    filtered_df = candidates_df[candidates_df["TFIDF_Similarity"] >= min_similarity]
    
    # Sort the remaining candidates by similarity (highest first)
    filtered_df = filtered_df.sort_values(by="TFIDF_Similarity", ascending=False)
    
    return filtered_df

def cluster_candidate_generation(cluster_history_ids, news_df, behaviors_df, cutoff_time, tokenizer, tfidf_vectorizer=None, min_tfidf_similarity=0.02, max_candidates=-1):

    # First, compute for each article the first interaction time from behaviors_df.
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    behaviors_df['Time'] = behaviors_df['Time'].apply(lambda t: t.tz_convert(None) if (t is not None and t.tzinfo is not None) else t)
    
    # Convert cutoff_time to datetime and remove timezone if present
    cutoff_time_dt = pd.to_datetime(cutoff_time)
    if cutoff_time_dt.tzinfo is not None:
        cutoff_time_dt = cutoff_time_dt.tz_convert(None)
    first_interactions = {}
    # Iterate over behaviors_df to compute the first interaction time for each news article
    for _, row in behaviors_df.iterrows():
        time_val = row["Time"]
        if pd.isna(time_val):
            continue
        if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
            continue
        for imp in row["Impressions"].split():
            try:
                art, label = imp.split('-')
                # Update the first interaction time if this is the earliest seen for the article
                if art not in first_interactions or time_val < first_interactions[art]:
                    first_interactions[art] = time_val
            except Exception as e:
                print(f"Error parsing impression {imp}: {e}")

    # Build candidate pool: articles with first interaction time <= cutoff_time
    candidate_pool = [art for art, t in first_interactions.items() if t <= cutoff_time_dt]
    # Remove articles that are already in the cluster history.
    candidate_pool = list(set(candidate_pool) - set(cluster_history_ids))
    # Build a candidate DataFrame from news_df using NewsID from news_df
    candidates_df = news_df[news_df['NewsID'].isin(candidate_pool)]

    # Optionally apply TF-IDF filtering using the cluster’s aggregated history text.
    if tfidf_vectorizer is not None:
        # Build the cluster history text from the cluster history IDs
        texts = []
        for art in cluster_history_ids:
            title_arr = news_df.loc[news_df['NewsID'] == art, 'CombinedText'].values
            if len(title_arr) > 0:
                texts.append(str(title_arr[0]))
        cluster_history_text = " ".join(texts)
        candidates_df = tfidf_filter_candidates(candidates_df, cluster_history_text, tfidf_vectorizer, min_similarity=min_tfidf_similarity)
    
    # Optionally limit the number of candidates.
    if max_candidates > 0 and len(candidates_df) > max_candidates:
        candidates_df = candidates_df.head(max_candidates)
    
    # For each candidate, create a tensor from its title (or CombinedText)
    candidate_tensors = []
    candidate_ids = []
    for idx, row in candidates_df.iterrows():
        art_id = row['NewsID']
        title = row['Title'] if pd.notna(row['Title']) and row['Title'].strip() != "" else " "
        seq = tokenizer.texts_to_sequences([title])
        if len(seq[0]) == 0:
            seq = [[0]]
        padded = pad_sequences(seq, maxlen=30, padding='post', truncating='post', value=0)[0]
        tensor = tf.convert_to_tensor([padded], dtype=tf.int32)
        candidate_tensors.append(tensor)
        candidate_ids.append(art_id)
    return candidate_tensors, candidate_ids

def cluster_rank_candidates(candidate_scores, candidate_ids, k):
    k = min(k, len(candidate_ids))  # ensure k doesn't exceed available candidates
    top_indices = np.argsort(candidate_scores)[-k:][::-1]
    logging.info(f"k:{k} candidate_ids: {len(candidate_ids)}!!!")
    for i in top_indices:
        logging.info(f"i:{i} candidate_ids[i]: {candidate_ids[i]}")
    recommended_ids = [candidate_ids[i] for i in top_indices]
    return recommended_ids

def average_precision_at_k(recommended_ids, ground_truth_ids, k, candidate_ids=None):
    if candidate_ids is None:
        candidate_ids = list(set(recommended_ids).union(ground_truth_ids))
    filtered_ground_truth = set(ground_truth_ids).intersection(set(candidate_ids))
    recommended_k = recommended_ids[:k]
    hit_count = 0
    sum_precisions = 0.0
    for i, rec in enumerate(recommended_k, start=1):
        if rec in filtered_ground_truth:
            hit_count += 1
            sum_precisions += hit_count / i
    return sum_precisions / hit_count if hit_count > 0 else 0.0

def compute_map_at_k(recommended_ids_list, ground_truth_list, k):
    ap_values = []
    for rec_ids, gt_ids in zip(recommended_ids_list, ground_truth_list):
        ap = average_precision_at_k(rec_ids, gt_ids, k)
        ap_values.append(ap)
    return np.mean(ap_values) if ap_values else 0.0

def dcg_at_k(recommended_ids, ground_truth_ids, k, candidate_ids=None):
    if candidate_ids is None:
        candidate_ids = list(set(recommended_ids).union(ground_truth_ids))
    filtered_ground_truth = set(ground_truth_ids).intersection(set(candidate_ids))
    recommended_k = recommended_ids[:k]
    dcg = 0.0
    for i, rec in enumerate(recommended_k, start=1):
        if rec in filtered_ground_truth:
            dcg += 1.0 / math.log2(i + 1)
    return dcg

def get_user_shown_articles(user_id, behaviors_df, cutoff_dt):
    shown_articles = set()
    logging.info(f"user_id:{user_id}")
    logging.info(f"behaviors_df:{behaviors_df}")
    logging.info(f"cutoff_dt:{cutoff_dt}")
    #behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    #cutoff_dt = pd.to_datetime(cutoff_time)
    future_rows = behaviors_df[
        (behaviors_df['UserID'] == user_id) &
        (behaviors_df['Time'] > np.datetime64(cutoff_dt))
    ]
    for _, row in future_rows.iterrows():
        if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
            continue
        for imp in row["Impressions"].split():
            try:
                art_id, _ = imp.split('-')
                shown_articles.add(art_id)
            except:
                pass
    return shown_articles

def get_or_compute_global_average_profile(behaviors_df, news_df, tokenizer, cutoff, filename='global_average_profile.pkl', 
                                            max_history_length=50, max_title_length=30):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            avg_profile = pickle.load(f)
        print(f"Loaded global average profile from {filename}")
        return avg_profile
    else:
        # Compute the global average profile.
        profiles = []
        user_ids = behaviors_df['UserID'].unique()
        total_users = len(user_ids)
        for i, user_id in enumerate(user_ids):
            history_tensor, history_ids, _ = build_user_profile_tensor(user_id, behaviors_df, news_df, cutoff, tokenizer,
                                                                       max_history_length, max_title_length)
            if history_ids:  # Include only users with a non-empty history.
                profiles.append(history_tensor.numpy())
            if i % 100 == 0:
                logging.info(f"{i+1}/{total_users} rows done.")
        if profiles:
            avg_profile = tf.convert_to_tensor(np.mean(np.stack(profiles, axis=0), axis=0), dtype=tf.int32)
        else:
            avg_profile = tf.zeros((max_history_length, max_title_length), dtype=tf.int32)
        # Save the computed global average profile.
        with open(filename, "wb") as f:
            pickle.dump(avg_profile, f)
        print(f"Saved global average profile to {filename}")
        return avg_profile

def build_user_profile_tensor(
    user_id,
    behaviors_df,
    news_df,
    cutoff_time_str, cutoff,
    tokenizer,
    avg_profile=None,
    pad_zeros=False,
    max_history_length=50,
    max_title_length=30,
    history_ids = None,
):

    if history_ids is None:
        behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
        user_hist = (
            behaviors_df
            .loc[(behaviors_df.UserID == user_id) &
                 (behaviors_df.Time <= cutoff)]
            .sort_values('Time')
        )
        full_ids = [
            art for txt in user_hist.History.str.split().dropna()
            for art in txt
        ]
    else:
        full_ids = list(history_ids)
    original_len = len(full_ids)
    history_article_ids = full_ids[-max_history_length:]

    # 2) tokenise
    seqs = tokenizer.texts_to_sequences(
        news_df.set_index('NewsID')
               .loc[history_article_ids, 'CombinedText']
               .fillna("")
               .tolist()
    )
    seqs = [
        s if s else [tokenizer.oov_token or 1]
        for s in seqs
    ]
    history_array = pad_sequences(
        seqs, maxlen=max_title_length, padding='post', truncating='post'
    )

    if pad_zeros and history_array.shape[0] < max_history_length:
        zeros = np.zeros(
            (max_history_length - history_array.shape[0], max_title_length),
            dtype=int
        )
        history_array = np.vstack([history_array, zeros])

    return tf.convert_to_tensor(history_array, dtype=tf.int32), \
           history_article_ids, original_len
    """
    original_history_len = len(full_ids)
    history_article_ids = full_ids[-max_history_length:]
    history_seqs = []
    for art_id in history_article_ids:
        row = news_df.loc[news_df['NewsID'] == art_id]
        text = row.iloc[0]['CombinedText'] if not row.empty else ""
        seq = tokenizer.texts_to_sequences([text])[0] or [0]
        seq = pad_sequences(
            [seq],
            maxlen=max_title_length,
            padding='post',
            truncating='post',
            value=0
        )[0]
        history_seqs.append(seq)

    history_array = np.array(history_seqs, dtype=int)

    if pad_zeros and history_array.shape[0] < max_history_length:
        pad_count = max_history_length - history_array.shape[0]
        zeros = np.zeros((pad_count, max_title_length), dtype=int)
        history_array = np.vstack([history_array, zeros])

    history_tensor = tf.convert_to_tensor(history_array, dtype=tf.int32)
    return history_tensor, history_article_ids, original_history_len
    """

def build_user_profiles(
    user_ids,
    behaviors_df,
    news_df,
    tokenizer,
    avg_profile,
    cutoff_time: str,
    max_history_length: int,
    max_title_length: int,
    resume=True,
    out_path: str = "user_profiles.pkl",
    cutoff=None
):
    profiles = {}
    progress_filename = "build_user_profiles_progress.pkl"
    if resume and os.path.exists(progress_filename):
        with open(progress_filename, "rb") as f:
            progress = pickle.load(f)
        logging.info(f"Resuming processing with progress: {progress}")
    else:
        progress = {}
    start_index = progress.get(out_path, 0)
    for i, user_id in enumerate(tqdm(user_ids[start_index:], desc=f"Building profiles for {out_path}"), start=start_index):
        hist_ids = get_user_history_ids(user_id, behaviors_df, cutoff)
        tensor, _, _ = build_user_profile_tensor(
            user_id=user_id,
            behaviors_df=behaviors_df,
            news_df=news_df,
            cutoff_time_str=cutoff_time,
            cutoff=cutoff,
            tokenizer=tokenizer,
            avg_profile=avg_profile,
            pad_zeros=False,
            max_history_length=max_history_length,
            max_title_length=max_title_length,
            history_ids=hist_ids
        )
        profiles[user_id] = (hist_ids, tensor)
        progress[out_path] = i + 1
        with open(progress_filename, "wb") as f:
            pickle.dump(progress, f)
    with open(out_path, "wb") as f:
        pickle.dump(profiles, f)
    return profiles

# Global cache for candidate pool for a given cutoff_time.
CANDIDATE_POOL_CACHE = {}

def precompute_candidate_pool(behaviors_df, cutoff_dt):
    # Convert times once
    if cutoff_dt.tzinfo is not None:
        cutoff_dt = cutoff_dt.tz_convert(None)
    
    first_interactions = {}
    for _, row in behaviors_df.iterrows():
        if pd.isna(row["Time"]) or pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
            continue
        if row["Time"] <= np.datetime64(cutoff_dt):
            for imp in row["Impressions"].split():
                art_id, _ = imp.split('-')
                # Record the earliest time seen for the article.
                if art_id not in first_interactions or row["Time"] < first_interactions[art_id]:
                    first_interactions[art_id] = row["Time"]
    candidate_pool = [art for art, t in first_interactions.items() if t <= cutoff_dt]
    return candidate_pool

def get_candidate_pool_for_user(user_id, behaviors_df, news_df, cutoff_time_str, cutoff, user_history_ids, remove_history=False):
    global CANDIDATE_POOL_CACHE
    if cutoff_time_str not in CANDIDATE_POOL_CACHE:
        CANDIDATE_POOL_CACHE[cutoff_time_str] = precompute_candidate_pool(behaviors_df, cutoff)
    candidate_pool = CANDIDATE_POOL_CACHE[cutoff_time_str]
    candidate_pool_user = list(set(candidate_pool))
    if remove_history:
        candidate_pool_user = list(set(candidate_pool) - set(user_history_ids))
    return candidate_pool_user

def user_candidate_generation(
    user_id,
    user_history_ids,
    behaviors_df,
    news_df,
    tokenizer,
    tfidf_vectorizer=None,
    cutoff_time_str=None,
    cutoff=None,
    min_tfidf_similarity=0.02,
    max_candidates=-1,
    max_title_length=30,
    id_to_index=None,title_tensor=None, exclude_clicked=False
):
    """
    Generating a candidate pool for a single user.
    1. gather all articles that exist up to cutoff_time.
    (2. Exclude articles in user_history_ids.)
    3. Optionally apply TF–IDF filter.
    4. Return candidate_tensors, candidate_ids.
    """

    logging.info(f"Filtering based on cutoff time: {cutoff_time_str}")
    if cutoff_time_str is not None:
        candidate_pool = get_candidate_pool_for_user(user_id, behaviors_df, news_df, cutoff_time_str, cutoff, user_history_ids)
    else:
        candidate_pool = news_df['NewsID'].unique().tolist()

    logging.info(f"After Filtering candidate pool length: {len(candidate_pool)}")
    # remove user's history from the candidate pool
    candidate_pool = list(set(candidate_pool))
    if exclude_clicked:
        candidate_pool = list(set(candidate_pool) - set(user_history_ids))
    logging.info(f"After removing user history from candidate pool: {len(candidate_pool)}")
    # Build candidates_df
    candidates_df = news_df[news_df['NewsID'].isin(candidate_pool)].copy()
    logging.info(f"built candidates_df")

    logging.info(f"tfidf filtering candidates when min_tfidf_similarity={min_tfidf_similarity}")
    if tfidf_vectorizer is not None and min_tfidf_similarity > 0.0:
        # Build user’s aggregated text from user_history_ids
        texts = []
        for art_id in user_history_ids:
            row_news = news_df[news_df['NewsID'] == art_id]
            if not row_news.empty:
                texts.append(str(row_news.iloc[0]['CombinedText']))
        user_history_text = " ".join(texts)

        candidates_df = tfidf_filter_candidates(candidates_df, user_history_text, tfidf_vectorizer, 
                                                    min_similarity=min_tfidf_similarity)
    logging.info(f"tfidf filtering done.")

    logging.info(f"filtering candidates (length:{len(candidates_df)}) with max size:{max_candidates}")
    if max_candidates > 0 and len(candidates_df) > max_candidates:
        candidates_df = candidates_df.head(max_candidates)

    logging.info(f"building candidate tensors")
    candidate_tensors = []
    candidate_ids = []
    for idx, row in candidates_df.iterrows():
        art_id = row['NewsID']
        title = row['Title'] if pd.notna(row['Title']) else ""
        seq = tokenizer.texts_to_sequences([title])[0]
        if len(seq) == 0:
            seq = [0]
        seq = pad_sequences([seq], maxlen=max_title_length, padding='post', truncating='post')[0]
        candidate_tensors.append(tf.convert_to_tensor([seq], dtype=tf.int32))  # shape (1, max_title_length)
        
        candidate_ids.append(art_id)
    
    #cand_indices = [id_to_index[nid] for nid in candidate_ids]
    #candidate_tensors = tf.gather(title_tensor, cand_indices)
    logging.info(f"lens: newsdf:{len(news_df['NewsID'].tolist())}candidate_pool:{len(candidate_pool)},candidate_pool:{len(candidate_pool)}")
    return candidate_tensors, candidate_ids

def get_user_future_clicks(user_id, behaviors_df, cutoff_time=None, cutoff_dt=None):
    user_future_clicks = set()
    if cutoff_dt == None:
        behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
        cutoff_dt = pd.to_datetime(cutoff_time)

    future_rows = behaviors_df[
        (behaviors_df['UserID'] == user_id)
        & (behaviors_df['Time'] > np.datetime64(cutoff_dt))
    ]
    for _, row in future_rows.iterrows():
        if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
            continue
        for imp in row["Impressions"].split():
            try:
                art_id, label = imp.split('-')
                label = int(label)
                if label == 1:  # means user actually clicked
                    user_future_clicks.add(art_id)
            except:
                pass

    return user_future_clicks
def compute_coverage(all_recommended_ids, total_num_articles):
    recommended_set = set()
    for rec_list in all_recommended_ids:
        recommended_set.update(rec_list)
    coverage_value = len(recommended_set) / total_num_articles
    return coverage_value

def write_partial_rows(rows, filename="user_level_partial_results.csv", results_dir='results'):
    if not rows:
        return
    df = pd.DataFrame(rows)
    f = f"{results_dir}/{filename}"
    file_exists = os.path.exists(f)

    df.to_csv(
        f,
        mode='a',
        header=not file_exists,  # write header if file doesn't exist
        index=False
    )

def score_candidates_in_batch(history_tensor, candidate_tensors, model, batch_size=128):

    num_candidates = len(candidate_tensors)
    if num_candidates == 0:
        return np.array([], dtype=float)

    # Stack all candidate rows => shape (num_candidates, max_title_length)
    batch_candidates = tf.concat(candidate_tensors, axis=0)

    # Repeat the user_history_tensor => shape (num_candidates, max_history_length, max_title_length)
    batch_history = tf.repeat(history_tensor, repeats=num_candidates, axis=0)

    preds = model.predict(
        {
           "history_input": batch_history,
           "candidate_input": batch_candidates
        },
        batch_size=batch_size
    )
    return preds.ravel()

def generate_base_predictions(history_tensor, candidate_tensors, models_dict, batch_size=128):
    separate_scores = {}
    for key, model in models_dict.items():
        history_arr = history_tensor.numpy()
        candidates_arr = np.vstack([t.numpy().squeeze(0) for t in candidate_tensors])
        preds = cached_score_single(history_arr, candidates_arr, key, batch_size)
        #preds = score_candidates_in_batch(history_tensor, candidate_tensors, model, batch_size)
        separate_scores[key] = preds  # shape (num_candidates,)
    return separate_scores

def save_incremental(prediction, filename='scores.pkl'):
    with open(filename, 'ab') as f:  # Open in append mode
        pickle.dump(prediction, f)

def score_candidates_ensemble_batch(history_tensor, candidate_tensors, models_dict, meta_models_dict, model_type, batch_size=128):

    all_preds = []
    separate_scores = {}
    separate_stacking_scores = {}
    separate_bagging_scores = {}
    cluster_keys = ['0_small','1_small','2_small','0_large','1_large','2_large']
    for key, model in models_dict.items():
        history_arr = history_tensor.numpy()
        candidates_arr = np.vstack([t.numpy().squeeze(0) for t in candidate_tensors])
        preds = cached_score_single(history_arr, candidates_arr, key, batch_size)
        #preds = score_candidates_in_batch(history_tensor, candidate_tensors, model, batch_size)
        all_preds.append(preds)
        separate_scores[key] = preds # Single model scores
        
    for key, meta_model in meta_models_dict.items():
        #key = model_type
        log_print(f"key:{key}")
        scores = {}
        if key == "all":
            scores = separate_scores
            all_preds = list(separate_scores.values())

        elif key == "cluster":
            for k, v in separate_scores.items():
                if k in cluster_keys or 'global' in k:
                    scores[k] = v
            log_print(f"scores:{scores}")
            all_preds = [v for k, v in separate_scores.items()
                    if k in cluster_keys or 'global' in k]
            log_print(f"all_preds:{all_preds}")

        elif key == "category":
            for k, v in separate_scores.items():
                if k not in cluster_keys:
                    scores[k] = v
            all_preds = [v for k, v in separate_scores.items()
                    if k not in cluster_keys]
        mean_preds = np.mean(all_preds, axis=0) # bagging scores
        separate_bagging_scores[key] = mean_preds

        if meta_model is not None:
            X_meta = build_meta_features(scores,None,None)
            meta_preds = meta_model.predict_proba(X_meta)[:, 1]
            separate_stacking_scores[key] = meta_preds
            log_print(f"X_meta:{X_meta}")
            log_print(f"meta_preds:{meta_preds}")


    return separate_bagging_scores, separate_scores, separate_stacking_scores

def candidate_pool_from_behavior(user_id, behaviors_df, cutoff_time=None, cutoff_dt=None):
    if cutoff_dt == None:
        behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
        cutoff_dt = pd.to_datetime(cutoff_time)
    pool = set()
    user_rows = behaviors_df[behaviors_df['UserID'] == user_id]
    for _, row in user_rows.iterrows():
        if row['Time'] > np.datetime64(cutoff_dt) and pd.notna(row["Impressions"]) and row["Impressions"].strip() != "":
            for imp in row["Impressions"].split():
                try:
                    art, _ = imp.split('-')
                    pool.add(art)
                except Exception as e:
                    print(f"Error parsing impression {imp}: {e}")
    return pool

def evaluate_candidate_pool(scores, candidate_ids, ground_truth_ids, k_values):
    # For a given candidate ids and corresponding scores, computing evaluation metrics.
    metrics = {}
    for k in k_values:
        effective_k = min(k, len(candidate_ids))
        if effective_k == 0:
            metrics[k] = {"recommended_ids": [],
                          "precision": 0.0,
                          "recall": 0.0,
                          "ap": 0.0,
                          "ndcg": 0.0,
                          "num_recommendations": 0}
            continue
        top_indices = np.argsort(scores)[-effective_k:][::-1]
        recommended_ids = [candidate_ids[i] for i in top_indices]
        prec, rec = compute_precision_recall_at_k(recommended_ids, ground_truth_ids, effective_k, candidate_ids)
        ap = average_precision_at_k(recommended_ids, ground_truth_ids, effective_k, candidate_ids)
        dcg_val = dcg_at_k(recommended_ids, ground_truth_ids, effective_k, candidate_ids)
        ideal_dcg = dcg_at_k(list(ground_truth_ids), ground_truth_ids, effective_k, candidate_ids)
        ndcg = dcg_val / ideal_dcg if ideal_dcg > 0 else 0.0
        metrics[k] = {
            "recommended_ids": recommended_ids,
            "precision": prec,
            "recall": rec,
            "ap": ap,
            "ndcg": ndcg,
            "num_recommendations": len(recommended_ids)
        }
    return metrics

def generate_results_row(scores, general_metrics, candidate_ids, behavior_candidate_ids, user_future_clicks, k_values, inc_full_metrics=True, auc_full=None, auc_behav=None):
    if inc_full_metrics:
        full_metrics = evaluate_candidate_pool(scores, candidate_ids, user_future_clicks, k_values)
    else:
        full_metrics = None
    
    # Compute metrics for the behavior-only candidate pool:
    # Filter scores and candidate_ids for those in the behavior-only pool.
    behavior_indices = [idx for idx, art in enumerate(candidate_ids) if art in behavior_candidate_ids]
    if behavior_indices:
        behavior_scores = [scores[idx] for idx in behavior_indices]
        behavior_candidate_ids_filtered = [candidate_ids[idx] for idx in behavior_indices]
        behavior_metrics = evaluate_candidate_pool(behavior_scores, behavior_candidate_ids_filtered, user_future_clicks, k_values)
    else:
        # If no behavior candidates, return zeros.
        behavior_metrics = {k: {"precision": 0.0, "recall": 0.0, "ap": 0.0, "ndcg": 0.0, "num_recommendations": 0} for k in k_values}
    
    # Prepare a result row for stacking ensemble metrics.
    partial_result_row = general_metrics
    for k in k_values:
        if full_metrics:
            partial_result_row[f"precision_full_{k}"] = full_metrics[k]["precision"]
            partial_result_row[f"recall_full_{k}"] = full_metrics[k]["recall"]
            partial_result_row[f"ap_full_{k}"] = full_metrics[k]["ap"]
            partial_result_row[f"ndcg_full_{k}"] = full_metrics[k]["ndcg"]
            partial_result_row[f"num_recommendations_full_{k}"] = full_metrics[k]["num_recommendations"]
        
        partial_result_row[f"precision_behavior_{k}"] = behavior_metrics[k]["precision"]
        partial_result_row[f"recall_behavior_{k}"] = behavior_metrics[k]["recall"]
        partial_result_row[f"ap_behavior_{k}"] = behavior_metrics[k]["ap"]
        partial_result_row[f"ndcg_behavior_{k}"] = behavior_metrics[k]["ndcg"]
        partial_result_row[f"num_recommendations_behavior_{k}"] = behavior_metrics[k]["num_recommendations"]
    if auc_full  is not None:
        partial_result_row["auc_full"]  = auc_full
    else:
        partial_result_row["auc_full"]  = ''
    if auc_behav is not None:
        partial_result_row["auc_behavior"] = auc_behav
    else:
        partial_result_row["auc_behavior"]  = ''
    partial_result_row["status"] = "DONE"

    return partial_result_row


def pad_history_with_avg_profile(filename=f"global_average_profile_large.pkl", user_history_tensor=None, max_history_length=50, user_id=None):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            avg_profile = pickle.load(f)
        print(f"Loaded global average profile from file:{avg_profile}")
        logging.info(f"Loaded global average profile from file:{avg_profile}")
    else:
        avg_profile = None
    if avg_profile is not None:
        if tf.reduce_sum(user_history_tensor) == 0:
            logging.info(f"User {user_id} has an empty history. Using global average profile:{avg_profile}")
            user_history_tensor = avg_profile
        elif tf.shape(user_history_tensor)[0] < max_history_length:
            # Pad the history tensor if it is too short.
            logging.info(f"User {user_id} has history:{user_history_tensor}")
            current_len = tf.shape(user_history_tensor)[0]
            missing = max_history_length - current_len
            padding = avg_profile[:missing, :]
            user_history_tensor = tf.concat([user_history_tensor, padding], axis=0)
            logging.info(f"User {user_id} has padded history:{user_history_tensor}")
        else:
            logging.info(f"user_history_tensor:{user_history_tensor}")
    return user_history_tensor

def _latest_checkpoint(meta_model_type: str,
                       model_type: str,
                       model_size: str,
                       encoder: bool = False,
                       ckpt_dir: str = "checkpoints/metamodels"):
    pattern = (f"{ckpt_dir}/{'encoder_' if encoder else ''}{meta_model_type}_meta_model_checkpoint_"
               f"{model_type}_{model_size}_batch_*.pkl")
    print(f"get checkpoint with pattern:{pattern}")
    paths = glob.glob(pattern)
    if not paths:
        return None
    print(paths)
    for p in paths:
        print(p, re.search(r"_batch_(\d+)\.pkl$", p))

    extract = lambda p: int(re.search(r"_batch_(\d+)\.pkl$", p).group(1))
    return max(paths, key=extract)

def lookup_categories(candidate_ids, news_df):
    series = (
        news_df.set_index("NewsID")
               .reindex(candidate_ids)["Category"]
               .fillna("Unknown")
    )
    candidate_categories = series.tolist()
    return candidate_categories

def get_user_history_ids(user_id,
                         behaviors_df,
                         cutoff_time):
    # Returns the list of all article IDs that user clicked on or saw before cutoff_time in chronological order.

    #df = behaviors_df.copy()
    #df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    cutoff = pd.to_datetime(cutoff_time)

    # filter behaviors for this user before cutoff
    user_df = behaviors_df[
        (behaviors_df['UserID'] == user_id) &
        (behaviors_df['Time'] <= cutoff)
    ].sort_values('Time')

    # Extract history IDs in order
    history_ids = []
    for txt in user_df['HistoryText'].fillna(''):
        if not txt.strip():
            continue
        for art in txt.split():
            if art not in history_ids:
                history_ids.append(art)

    return history_ids

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

def build_meta_features(separate_scores: dict[str, np.ndarray],
                        candidate_categories: list[str] | None = None,
                        encoder: OneHotEncoder | None = None) -> np.ndarray:
    X_base = np.column_stack([separate_scores[k] for k in sorted(separate_scores)])
    if candidate_categories is not None and encoder is not None:
        cat_mat = encoder.transform(np.array(candidate_categories).reshape(-1, 1))
        X_base = np.hstack([X_base, cat_mat])
    return X_base.astype("float32")

def evaluate_and_buffer_user(
    user_id, cluster_id, user_index,
    behaviors_df, news_df, cutoff_time_str, cutoff, tokenizer,
    avg_profile, max_history_length,
    train_data, test_data, tfidf_vectorizer,
    models_dict,
    meta_models_dict,
    model_type,
    hist_ids,
    pivot: int,
    partial_buffer=None, k_values=None, partial_csv="out.csv",
    donor_id = None,
    pad_zeros=True,
    drift_fraction=0,user_profiles=None,id_to_index=None,title_tensor=None,
    use_full_avg_profile=False
):
    # Pads / substitutes the user_history_tensor, builds candidates, scores them with bagging, stacking, and individual models,
    # then writes partial CSV rows.
    logging.debug(f'User:{user_id}, First 5 hist IDs received: {hist_ids[:5]}')
    shown = get_user_shown_articles(user_id, behaviors_df, cutoff)
    
    if len(shown) == 0:
        logging.info(f"shown is empty")
        candidate_ids = []
        num_candidates = len(candidate_ids)
        zero_scores = np.zeros(num_candidates, dtype=float)
        separate_zero = {k: zero_scores.copy() for k in models_dict}
        behaviour_ids = []
        general_metrics = {
            "cluster_id": cluster_id,
            "user_id": user_id,
            "user_index_in_cluster": user_index,
            "donor_user_id": donor_id,
            "num_from_original": 0,
            "num_from_donor": 0,
            "pivot": pivot,
            "num_history_articles": 0,
            "original_history_len": 0,
            "num_future_clicks":    0,
            "num_shown_articles":   0,
            "num_candidates_full":  num_candidates,
            "num_candidates_behavior": len(behaviour_ids),
            "num_ground_truth_all":  0,
            "num_ground_truth_in_behavior": 0,
        }
        # buffer zero‐score rows
        for key in ("bagging", "stacking"):
            partial_buffer[key].append(
                generate_results_row(zero_scores, general_metrics, candidate_ids, behaviour_ids, [], k_values)
            )
            write_partial_rows(partial_buffer[key], f"{key}_{partial_csv}")
            partial_buffer[key].clear()
        for model_key in models_dict:
            partial_buffer[model_key].append(
                generate_results_row(separate_zero[model_key], general_metrics, candidate_ids, behaviour_ids, [], k_values)
            )
            write_partial_rows(partial_buffer[model_key], f"{model_key}_{partial_csv}")
            partial_buffer[model_key].clear()
        return
    logging.info(f"shown contains:{shown}")

    if use_full_avg_profile:
        user_history_tensor = avg_profile
        used_ids = []
        original_history_len = 0
    else:
        user_history_tensor, used_ids, original_history_len = build_user_profile_tensor(
            user_id=user_id,
            behaviors_df=behaviors_df,
            news_df=news_df,
            cutoff_time_str=cutoff_time_str,
            cutoff=cutoff,
            tokenizer=tokenizer,
            avg_profile=avg_profile,
            pad_zeros=pad_zeros,
            max_history_length=max_history_length,
            max_title_length=30,
            history_ids=hist_ids
        )
    num_history_articles = len(used_ids)
    num_from_original = min(pivot, len(used_ids))
    num_from_donor    = num_history_articles - num_from_original

    candidate_tensors, candidate_ids = user_candidate_generation(
        user_id, hist_ids, test_data, news_df,
        tokenizer, tfidf_vectorizer, cutoff_time_str, cutoff, 0.0,id_to_index=id_to_index,title_tensor=title_tensor)
    num_candidates = len(candidate_ids)
    if avg_profile is not None:
        if tf.reduce_sum(user_history_tensor) == 0:
            logging.info(f"User {user_id} empty, using global avg")
            user_history_tensor = avg_profile
        # too short to pad with avg vectors
        elif tf.shape(user_history_tensor)[0] < max_history_length:
            curr = tf.shape(user_history_tensor)[0]
            miss = max_history_length - curr
            padding = avg_profile[:miss, :]
            logging.info(f"Padding user {user_id} history from {curr} to {max_history_length}")
            user_history_tensor = tf.concat([user_history_tensor, padding], axis=0)
        # final check
        if tf.reduce_sum(user_history_tensor) == 0:
            logging.info(f"After pad, user {user_id} still zero, use full global avg")
            user_history_tensor = avg_profile
    log_print(f"user_history_tensor:{user_history_tensor}")
    separate_bagging_scores, separate_scores, separate_stacking_scores = score_candidates_ensemble_batch(
        tf.expand_dims(user_history_tensor, 0),
        candidate_tensors,
        models_dict,
        meta_models_dict,
        batch_size=512,
        model_type=model_type
    )
    """
    if meta_model is not None:
        X_meta = build_meta_features(
                    separate_scores,
                    lookup_categories(candidate_ids, news_df)
                        if encoder is not None else None,
                    encoder)
        meta_preds = meta_model.predict_proba(X_meta)[:, 1]
        separate_scores["meta"] = meta_preds
    """
    logging.info(f"bagging_scores={separate_bagging_scores}, separate_scores={separate_scores}, separate_stacking_scores={separate_stacking_scores}")

    user_future = get_user_future_clicks(user_id, test_data, cutoff_dt=cutoff)
    behaviour_ids = list(candidate_pool_from_behavior(user_id, test_data, cutoff_dt=cutoff))


    labels_full = np.asarray([1 if cid in user_future else 0 for cid in candidate_ids], dtype=int)
    mask_beh = np.isin(candidate_ids, behaviour_ids)
    labels_beh = labels_full[mask_beh]

    general_metrics = {
        "cluster_id": cluster_id,
        "user_id": user_id,
        "user_index_in_cluster": user_index,
        "donor_user_id": donor_id,
        "num_from_original": num_from_original,
        "num_from_donor": num_from_donor,
        "pivot": pivot,
        "num_history_articles": num_history_articles,
        "original_history_len": original_history_len,
        "num_future_clicks":    len(user_future),
        "num_shown_articles":   len(shown),
        "num_candidates_full":  len(candidate_ids),
        "num_candidates_behavior": len(behaviour_ids),
        "num_ground_truth_all":     len(user_future),
        "num_ground_truth_in_behavior": len(set(user_future) & set(behaviour_ids)),
    }
    for key, bagging_scores in separate_bagging_scores.items():
        if f"bagging_{key}" not in partial_buffer:
            partial_buffer[f"bagging_{key}"] = []
        partial_buffer[f"bagging_{key}"].append(
            generate_results_row(bagging_scores, general_metrics, candidate_ids, behaviour_ids, user_future, k_values, auc_full = safe_auc(labels_full, bagging_scores), auc_behav  = safe_auc(labels_beh,  bagging_scores[mask_beh]))
        )
        write_partial_rows(partial_buffer[f"bagging_{key}"], f"bagging_{key}_{partial_csv}")
        partial_buffer[f"bagging_{key}"].clear()

    for key, stacking_scores in separate_stacking_scores.items():
        if f"stacking_{key}" not in partial_buffer:
            partial_buffer[f"stacking_{key}"] = []
        partial_buffer[f"stacking_{key}"].append(
            generate_results_row(stacking_scores, general_metrics, candidate_ids, behaviour_ids, user_future, k_values, auc_full = safe_auc(labels_full, stacking_scores), auc_behav  = safe_auc(labels_beh,  stacking_scores[mask_beh]))
        )
        write_partial_rows(partial_buffer[f"stacking_{key}"], f"stacking_{key}_{partial_csv}")
        partial_buffer[f"stacking_{key}"].clear()

    for model_key in models_dict:
        partial_buffer[model_key].append(
            generate_results_row(separate_scores[model_key], general_metrics, candidate_ids, behaviour_ids, user_future, k_values, auc_full = safe_auc(labels_full, separate_scores[model_key]), auc_behav  = safe_auc(labels_beh,  separate_scores[model_key][mask_beh]))
        )
        #save_incremental(separate_scores, f"{model_key}_{partial_csv.split('.csv')[0]}.pkl")
        write_partial_rows(partial_buffer[model_key], f"{model_key}_{partial_csv}")
        partial_buffer[model_key].clear()

def make_swapped_history(
    user_id,
    real_history_ids,
    behaviors_df,
    cutoff,
    viable_donors,
    drift_fraction: float = 0.5,
    max_history_length = 50,
    strategy='random',
    head_swap: bool = True
):
    N = min(max_history_length,len(real_history_ids))
    pivot = max(1, int(N * drift_fraction))
    user_seed = hash(user_id) % 2**32
    user_rng = np.random.default_rng(user_seed)

    donor = user_rng.choice(viable_donors) # Random donor
    # get full donor history up to cutoff, then truncate to match N
    donor_full = get_user_history_ids(donor, behaviors_df, cutoff)
    #log_print(f"donor_full:{donor_full}")
    donor_trunc = donor_full[-N:]

    if head_swap:
        # use donor for the first pivot, original for the remainder
        pre_ids   = donor_trunc[:pivot]
        post_ids   = real_history_ids[pivot:N]
    else:
        # use original for the first (N−pivot), donor for the tail
        pre_ids   = real_history_ids[: N-pivot]
        post_ids   = donor_trunc[N-pivot : N]
    swapped_ids = pre_ids + post_ids

    return swapped_ids, pivot, donor

def get_meta_models(meta_model_type, model_type, model_size, date_str):
    meta_models = {}
    encoders = {}
    model_types = [model_type]
    if model_type == "all":
        model_types.append("cluster")
        model_types.append("category")
    for model_type in model_types:
        meta_model_path = (
            f"{meta_model_type}_latest_meta_model_"
            f"{model_type}_{model_size}_{date_str}.pkl"
        )
        encoder_meta_model_path = (
            f"encoder_{meta_model_type}_latest_meta_model_"
            f"{model_type}_{model_size}_{date_str}.pkl"
        )

        try:
            with open(meta_model_path, "rb") as fh:
                meta_model = pickle.load(fh)
                print(f"Loaded final meta‑model from {meta_model_path}")
            with open(encoder_meta_model_path, "rb") as fh:
                encoder = pickle.load(fh)
                print(f"Loaded final meta‑model from {encoder_meta_model_path}")
        except FileNotFoundError:
            ckpt_path = _latest_checkpoint(meta_model_type, model_type, model_size)
            enc_ckpt_path = _latest_checkpoint(meta_model_type, model_type, model_size, encoder=True)
            print(f"enc_ckpt_path{enc_ckpt_path}")
            if ckpt_path:
                with open(ckpt_path, "rb") as fh:
                    meta_model = pickle.load(fh)
                print(f"Final model absent – resumed from checkpoint {ckpt_path}")
            else:
                meta_model = None
                raise FileNotFoundError(
                    f"No meta model found at {meta_model_path} and no checkpoints found from {ckpt_path}"
                    f"in {os.path.join('checkpoints', 'metamodels')}"
                )
            if enc_ckpt_path:
                with open(enc_ckpt_path, "rb") as fh:
                    encoder = pickle.load(fh)
                print(f"Final model absent – resumed from checkpoint {enc_ckpt_path}")
            else:
                encoder = None
                raise FileNotFoundError(
                    f"No meta model found at {encoder_meta_model_path} and no checkpoints found from {enc_ckpt_path}"
                    f"in {os.path.join('checkpoints', 'metamodels')}"
                )
        meta_models[model_type] = meta_model
        encoders[model_type] = encoder
    return meta_models, encoders

def run_experiments_user_level(cluster_mapping, viable_donors, train_data, test_data, news_df,
                                    behaviors_df, models_dict, tokenizer,
                                    tfidf_vectorizer, cutoff_time_str, cutoff, 
                                    k_values=[5, 10, 20, 50],
                                    partial_csv="user_level_partial_results.csv",
                                    shuffle_clusters=False, cluster_order=None,
                                    meta_model_type='SGDClassifier', model_type='cluster',
                                    dataset_size='large',model_size='large',date_str='latest',resume=True,
                                    adaptivity_test=False,drift_fraction=0.5,donor_strategy='random',
                                    rebuild_after_drift=False,measure_warm_up=3,seed=42,shuffle_user=False,avg_profile=None,         
                                    user_profiles=None,id_to_index=None,title_tensor=None,
                                    meta_model_path = "meta/meta_model_XGBClassifier_hist_cluster_small_train_small.pkl",
                                    encoder_path = None,meta_model_pattern=None, pad_zeros=True, use_full_avg_profile=False):
    logging.info(f"partial_csv: {partial_csv}, viable_donors:{viable_donors}, models_dict:{models_dict}, model_type:{model_type}, dataset_size:{dataset_size}, model_size:{model_size}")
    results = []
    total_articles = len(news_df)
    rng = np.random.default_rng(seed)

    # Initialize a partial buffer for ensemble (bagging and stacking) and for each individual model.
    partial_buffer = {"bagging": [], "stacking": []}
    for model_key in models_dict.keys():
        partial_buffer[model_key] = []

    if cluster_order is not None:
        ordered_clusters = cluster_order
    else:
        ordered_clusters = list(cluster_mapping.keys())
        if shuffle_clusters:
            np.random.shuffle(ordered_clusters)

    progress_filename = "user_level_progress.pkl"
    if drift_fraction != 0.5: # non default value
        progress_filename = f"user_level_progress_{drift_fraction}.pkl"
    if resume and os.path.exists(progress_filename):
        with open(progress_filename, "rb") as f:
            progress = pickle.load(f)
        logging.info(f"Resuming processing with progress: {progress}")
    else:
        progress = {}

    #if model_type != "global": # get meta model
    #    meta_models, encoders = get_meta_models(meta_model_type, model_type, model_size, date_str)
    #logging.info(f"meta_models:{meta_models}, encoders:{encoders}")

    """
    from precompute_cache import MAX_TITLE_LEN, MAX_HISTORY
    seqs  = tokenizer.texts_to_sequences(news_df['Title'].fillna("").tolist())
    padded = pad_sequences(seqs, maxlen=max_title_length, padding='post')
    np.savez_compressed("title_tensor.npz",
                        tensor=padded.astype('int32'),
                        id2idx=news_df['NewsID'].to_numpy())
    title_npz   = np.load("title_tensor.npz")
    title_tensor = tf.convert_to_tensor(title_npz["tensor"], dtype=tf.int32)
    id2idx      = {nid: i for i, nid in enumerate(title_npz["ids"])}

    profiles     = pickle.load(open("user_prof.pkl", "rb"))
    tfidf_masks  = np.load("user_tfidf_masks.npz")  
    def build_candidate_tensor(uid):
        hist_ids, _ = profiles[uid]
        art_mask    = tfidf_masks[uid]                        # bool vector len N_articles
        cand_idx    = np.flatnonzero(art_mask & ~np.isin(title_npz["ids"], hist_ids))
        cand_idx    = cand_idx[:MAX_CAND]                    # cap size
        return tf.gather(title_tensor, cand_idx), cand_idx

    def dataset_generator(user_ids):
        for uid in user_ids:
            hist_ids, hist_tensor = profiles[uid]
            cand_tensor, cand_idx = build_candidate_tensor(uid)
            yield hist_tensor, cand_tensor, cand_idx, uid      # uid for bookkeeping

    user_ids = [u for cluster in cluster_mapping.values() for u in cluster]
    ds = tf.data.Dataset.from_generator(
            lambda: dataset_generator(user_ids),
            output_signature=(
                tf.TensorSpec(shape=(MAX_HISTORY, MAX_TITLE_LEN), dtype=tf.int32),
                tf.TensorSpec(shape=(None, MAX_TITLE_LEN),    dtype=tf.int32),
                tf.TensorSpec(shape=(None,),                  dtype=tf.int32),
                tf.TensorSpec(shape=(),                       dtype=tf.string)
            )
        ).batch(32).prefetch(tf.data.AUTOTUNE)
    """
    meta_models_dict = {}
    log_print(f"meta_model_pattern:{meta_model_pattern}")

    if meta_model_pattern is not None:
        if meta_model_pattern.count('*') != 1:
            raise ValueError("meta_model_pattern should contain 1 '*'")
        prefix, suffix = meta_model_pattern.split('*', 1)
        log_print(f"prefix:{prefix}")
        log_print(f"suffix:{suffix}")
        for path in Path().glob(meta_model_pattern):
            log_print(f"{path}")
            if not (f"{path}".startswith(prefix) and f"{path}".endswith(suffix)):
                continue
            key = f"{path}"[len(prefix):-len(suffix)] if suffix else f"{path}"[len(prefix):]
            log_print(f"key:{key}")
            meta_models_dict[key] = joblib.load(f"{path}")
    log_print(f"meta_models_dict:{meta_models_dict}")
    for cluster_id in ordered_clusters:
        user_list = cluster_mapping[cluster_id]
        if shuffle_user:
            rng.shuffle(user_list)
        total_users = len(user_list)
        start_index = progress.get(cluster_id, 0)
        for i, user_id in enumerate(tqdm(user_list[start_index:], desc=f"Evaluating users in cluster {cluster_id}"), start=start_index):
            try:
                logging.info(f"Starting user {user_id} in cluster {cluster_id} (index {i})")

                real_history_ids = get_user_history_ids(user_id, behaviors_df, cutoff)
                evaluate_and_buffer_user(
                    user_id, cluster_id, i,
                    behaviors_df, news_df, cutoff_time_str, cutoff, tokenizer,
                    avg_profile, max_history_length,
                    train_data, test_data, tfidf_vectorizer,
                    models_dict=models_dict,meta_models_dict=meta_models_dict,model_type=model_type,
                    partial_buffer=partial_buffer, k_values=k_values,
                    partial_csv=partial_csv, hist_ids=real_history_ids,
                    pivot=len(real_history_ids), donor_id=None,
                    user_profiles=user_profiles,
                    id_to_index=id_to_index, title_tensor=title_tensor,
                    pad_zeros=pad_zeros,
                    use_full_avg_profile=use_full_avg_profile
                )

                logging.info(f"user_id={user_id},real_history_ids:{real_history_ids}")
                if adaptivity_test:
                    swapped_ids, pivot, donor = make_swapped_history(user_id, real_history_ids, behaviors_df, cutoff, viable_donors, drift_fraction, max_history_length, donor_strategy)

                    logging.info(f"user_id={user_id},swapped_ids:{swapped_ids}")

                    evaluate_and_buffer_user(
                        user_id, cluster_id, i,
                        behaviors_df, news_df, cutoff_time_str, cutoff, tokenizer,
                        avg_profile, max_history_length,
                        train_data, test_data, tfidf_vectorizer,
                        models_dict=models_dict,meta_models_dict=meta_models_dict,model_type=model_type,
                        partial_buffer=partial_buffer, k_values=k_values,
                        partial_csv   = partial_csv,
                        hist_ids      = swapped_ids,
                        pivot         = pivot,
                        donor_id      = donor,
                        drift_fraction=drift_fraction,
                        user_profiles=user_profiles,
                        id_to_index=id_to_index,
                        title_tensor=title_tensor,
                        pad_zeros=pad_zeros,
                        use_full_avg_profile=use_full_avg_profile
                    )
                    
            except Exception as e:
                logging.error(f"Failed on user {user_id} in cluster {cluster_id} with error: {e}\n{format_exc()}")
                partial_buffer["bagging"].append({
                    "cluster_id": cluster_id,
                    "user_id": user_id,
                    "user_index_in_cluster": i,
                    "status": f"FAILED: {e}"
                })
                continue
            progress[cluster_id] = i + 1
            with open(progress_filename, "wb") as f:
                pickle.dump(progress, f)

        for key in partial_buffer:
            if partial_buffer[key]:
                write_partial_rows(partial_buffer[key], f"{model_size}_{partial_csv}")
                partial_buffer[key] = []

    results_df = pd.DataFrame(results)
    results_df.to_csv("user_level_experiment_results.csv", index=False)
    print("Cluster-level experiment results saved to 'user_level_experiment_results.csv'")
    return results_df

def transform_history(hist_ids, *, strategy="full",
                      k=50, donor_ids=None, pivot=5):
    if strategy == "truncate":
        return hist_ids[-k:]
    if strategy == "remove":
        return []
    if strategy == "swap":
        if donor_ids is None:
            raise ValueError("donor_ids required for swap")
        tail_len  = max(0, len(hist_ids) - pivot)
        return donor_ids[-pivot:] + hist_ids[-tail_len:]
    return hist_ids

def get_donor_history(target_uid, impression_time, behaviors_df,
                      max_len=50, rng=np.random.default_rng()):
    cand_mask = (behaviors_df['Time'] <= impression_time) & \
                (behaviors_df['UserID'] != target_uid)
    donors = behaviors_df.loc[cand_mask, 'UserID'].unique()
    if donors.size == 0:
        return []

    donor_uid = rng.choice(donors)

    donor_rows = behaviors_df[
        (behaviors_df['UserID'] == donor_uid) &
        (behaviors_df['Time'] <= impression_time)
    ]
    donor_clicks = []
    for h in donor_rows['HistoryText']:
        if pd.notna(h):
            donor_clicks.extend(h.split())

    return donor_clicks[-max_len:]

def pad_or_trunc(
        news_ids: list[str],
        news_text_dict: dict[str, list[int]],
        max_history_length: int,
        max_title_length: int
    ) -> list[list[int]]:

    # look up every title; fall back to all-zeros if a NewsID is missing
    title_tokens = [
        news_text_dict.get(nid, [0] * max_title_length)[:max_title_length]
        for nid in news_ids[-max_history_length:]
    ]
    # left-pad with rows of zeros so that len == max_history_length
    pad_rows = max_history_length - len(title_tokens)
    if pad_rows > 0:
        title_tokens = [[0]*max_title_length]*pad_rows + title_tokens
    return title_tokens

def make_eval_sets_from_train(
        df: pd.DataFrame,
        pivots: list[int] = (45,),
        ks:     list[int] = (5,),
        frac:   float     = 1.0,
        dataset: str      = "valid",
        dataset_size: str = "small",
        out_dir:  str     = ".",
    ):
    print(f"columns in original df:{df.columns}")
    """
    Build eval-frames that look exactly like train_df and store them as pickles.
    --------------------------------------------------------------------------
    * frac    – sample only a fraction to speed up experimentation
    * pivots  – sizes for the 'swap' history variant
    * ks      – last-k history variants
    """
    os.makedirs(out_dir, exist_ok=True)
    if frac < 1.0:
        subset_tag = f"_{frac}"
        #base = sample_behaviors(df, frac)
    else:
        subset_tag = ""
        #base = df
    base = df.sample(frac=frac, random_state=0).reset_index(drop=True)

    ### history variants ----------------------------------------------------
    rng = np.random.default_rng(0)
    donor_histories = base["HistoryTitles"].sample(frac=1, random_state=1).tolist()

    variants: dict[str, list] = {"Hist_full": base["HistoryTitles"]}

    for k in ks:
        variants[f"Hist_k{k}"] = base["HistoryTitles"].apply(lambda h: h[-k:])

    for p in pivots:
        variants[f"Hist_swap{p}"] = [
            donor[-p:] + hist[-(len(hist) - p):]
            for donor, hist in zip(donor_histories, base["HistoryTitles"])
        ]
    def _fname(stem):
        return (f"{out_dir}/evaluation_{dataset}_{dataset_size}"
                f"{subset_tag}_{stem}{subset_tag}.pkl")
    ### write out one DataFrame per variant ---------------------------------
    keep = ["UserID", "CandidateTitleTokens", "Label", "ImpressionID"]
    for name, hist_col in variants.items():
        out = base[keep].copy()
        out["HistoryTitles"] = hist_col
        #fname = f"{out_dir}/evaluation_{dataset}_{dataset_size}_{_fname(name)}.pkl"
        
        print(f"columns in out df:{out.columns}")
        print(f"writing to {_fname(name)}")
        out.to_pickle(_fname(name), protocol=4)   # ⚑ fastest & smallest for nested lists

def load_full_news(train_dir, dev_dir, dev_dir_big = 'dataset/valid/'):
    cols = ['NewsID','Category','SubCategory','Title','Abstract',
            'URL','TitleEntities','AbstractEntities']
    print(f"loading from {train_dir}")
    news_train = pd.read_csv(os.path.join(train_dir, 'news.tsv'), sep='\t',
                names=cols, index_col=False)
    print(f"loaded news {news_train}")
    print(f"loading from {dev_dir}")
    news_dev   = pd.read_csv(os.path.join(dev_dir,   'news.tsv'), sep='\t',
                names=cols, index_col=False)
    print(f"loaded news {news_dev}")
    print(f"loading from {dev_dir_big}")
    news_dev_big   = pd.read_csv(os.path.join(dev_dir_big,   'news.tsv'), sep='\t',
                names=cols, index_col=False)
    print(f"loaded news {news_dev_big}")
    news_train['CleanTitle'] = news_train['Title'].apply(clean_text)
    news_train['CleanAbstract'] = news_train['Abstract'].apply(clean_text)
    news_train['CombinedText'] = news_train['CleanTitle'] + ' ' + news_train['CleanAbstract']
    news_train["CombinedText"] = news_train["CombinedText"].astype(str)
    news_train["CombinedText"] = news_train["CombinedText"].fillna("")

    news_dev['CleanTitle'] = news_dev['Title'].apply(clean_text)
    news_dev['CleanAbstract'] = news_dev['Abstract'].apply(clean_text)
    news_dev['CombinedText'] = news_dev['CleanTitle'] + ' ' + news_dev['CleanAbstract']
    news_dev["CombinedText"] = news_dev["CombinedText"].astype(str)
    news_dev["CombinedText"] = news_dev["CombinedText"].fillna("")

    news_dev_big['CleanTitle'] = news_dev_big['Title'].apply(clean_text)
    news_dev_big['CleanAbstract'] = news_dev_big['Abstract'].apply(clean_text)
    news_dev_big['CombinedText'] = news_dev_big['CleanTitle'] + ' ' + news_dev_big['CleanAbstract']
    news_dev_big["CombinedText"] = news_dev_big["CombinedText"].astype(str)
    news_dev_big["CombinedText"] = news_dev_big["CombinedText"].fillna("")
    news_full  = (pd.concat([news_train, news_dev, news_dev_big], ignore_index=True)
                .drop_duplicates('NewsID'))
    return news_full

def prepare_train_df(
    data_dir,
    train_data_dir,
    valid_data_dir,
    news_file,
    behaviors_file,
    user_category_profiles,
    num_clusters=3,
    fraction=1,
    max_title_length=30,
    max_history_length=50,
    downsampling=False,
    categorized_samples=False,
    news_df_pkl="models/news_df_processed.pkl", train_df_pkl="models/train_df_processed.pkl",
    test_size=0.2, split_indepently=False, dataset="train", dataset_size="small", process_valid_sets=False, behavior_pickle = "",
    eval_frac=1.0, big_tokenizer=False, pivots=[45], ks=[5]
    ):
    log_print(f"""data_dir={data_dir},
    valid_data_dir={valid_data_dir},
    news_file={news_file},
    behaviors_file={behaviors_file},
    user_category_profiles={user_category_profiles},
    downsampling={downsampling},
    categorized_samples={categorized_samples},
    news_df_pkl={news_df_pkl}, train_df_pkl={train_df_pkl},
    test_size={test_size}""")
    news_full = load_full_news(train_data_dir, valid_data_dir)
    if dataset == "valid":
        data_dir=valid_data_dir
    print(f"loading data from dir data_dir:{data_dir}")
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    train_news_path = os.path.join(train_data_dir, news_file)
    train_news_df = pd.read_csv(
        train_news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    print("Loaded news data:")
    print(news_df.head())

    behaviors_path = os.path.join(data_dir, behaviors_file)
    if behavior_pickle == "":
        behaviors_df = pd.read_csv(
            behaviors_path,
            sep='\t',
            names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
            index_col=False
        )
    else:
        behaviors_df = pd.read_pickle(behavior_pickle)
    print("Loaded behaviors data:")
    print(behaviors_df.head())

    news_df['CleanTitle'] = news_df['Title'].apply(clean_text)
    news_df['CleanAbstract'] = news_df['Abstract'].apply(clean_text)
    news_df['CombinedText'] = news_df['CleanTitle'] + ' ' + news_df['CleanAbstract']
    news_df["CombinedText"] = news_df["CombinedText"].astype(str)
    news_df["CombinedText"] = news_df["CombinedText"].fillna("")

    # Fit the tokenizer as it was fitted for model training!!!!!!!!!! Tokenizer needs to be exact same as it was during training to match mapping
    train_news_df['CleanTitle'] = train_news_df['Title'].apply(clean_text)
    train_news_df['CleanAbstract'] = train_news_df['Abstract'].apply(clean_text)
    train_news_df['CombinedText'] = train_news_df['CleanTitle'] + ' ' + train_news_df['CleanAbstract']
    train_news_df["CombinedText"] = train_news_df["CombinedText"].astype(str)
    train_news_df["CombinedText"] = train_news_df["CombinedText"].fillna("")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_news_df["CombinedText"].tolist())

    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")
    with open('train_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    """
    padded = pad_sequences(
        tokenizer.texts_to_sequences(news_df["CombinedText"]),
        maxlen=30, padding='post', truncating='post'
    )
    news_df["PaddedText"] = padded.tolist()
    """
    news_df['EncodedText'] = tokenizer.texts_to_sequences(news_df['CombinedText'])
    news_df['PaddedText'] = list(pad_sequences(news_df['EncodedText'], maxlen=max_title_length, padding='post', truncating='post'))
    #news_full["PaddedText"] = pad_sequences(
    #    tokenizer.texts_to_sequences(news_full["CombinedText"]),
    #    maxlen=30, padding='post', truncating='post')
    news_text_dict = dict(zip(news_df["NewsID"], news_df["PaddedText"]))

    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    news_df['EncodedText'] = tokenizer.texts_to_sequences(news_df['CombinedText'])
    news_df['PaddedText'] = list(pad_sequences(news_df['EncodedText'], maxlen=max_title_length, padding='post', truncating='post'))
    news_text_dict = dict(zip(news_df['NewsID'], news_df['PaddedText']))
    """

    def parse_impressions(impressions):
        impression_list = impressions.split()
        news_ids = []
        labels = []
        for imp in impression_list:
            try:
                news_id, label = imp.split('-')
                news_ids.append(news_id)
                labels.append(int(label))
            except ValueError:
                continue
        return news_ids, labels

    # Parse news ids and labels behaviors data
    log_print(f"Parsing 'ImpressionNewsIDs', 'ImpressionLabels'")
    behaviors_df[['ImpressionNewsIDs', 'ImpressionLabels']] = behaviors_df['Impressions'].apply(
        lambda x: pd.Series(parse_impressions(x))
    )

    train_samples = []
    log_print(f"Creating category map")
    category_map = dict(zip(news_df['NewsID'], news_df['Category']))
    #candidate_category_series = news_df[news_df['NewsID'] == candidate_news_id]['Category']
    # Iterate over behaviors to create train samples
    # for row in tqdm(behaviors_df.itertuples(index=False), total=len(behaviors_df)):
    for _, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0]):
        user_id = row['UserID']
        #user_cluster = row['Cluster'] if 'Cluster' in row else None

        # Parse user history
        history_ids = row['HistoryText'].split() if pd.notna(row['HistoryText']) else []
        history_texts = [news_text_dict.get(nid, [0]*max_title_length) for nid in history_ids]

        # Limit history length
        if len(history_texts) < max_history_length:
            padding = [[0]*max_title_length] * (max_history_length - len(history_texts))
            history_texts = padding + history_texts
        else:
            history_texts = history_texts[-max_history_length:]

        candidate_news_ids = row['ImpressionNewsIDs']
        labels = row['ImpressionLabels']
        impr_id = row['ImpressionID']
        for candidate_news_id, label in zip(candidate_news_ids, labels):
            candidate_text = news_text_dict.get(candidate_news_id, [0]*max_title_length)
            sample = {
                'ImpressionID'        : impr_id,
                'UserID': user_id,
                'HistoryTitles': history_texts,
                'CandidateTitleTokens': candidate_text,
                'Label': label
            }
            if categorized_samples:
                #candidate_category_series = news_df[news_df['NewsID'] == candidate_news_id]['Category']
                #candidate_category = candidate_category_series.iloc[0] if not candidate_category_series.empty else "Unknown"
                candidate_category = category_map.get(candidate_news_id, "Unknown")
                sample['CandidateCategory'] = candidate_category
            train_samples.append(sample)

    train_df = pd.DataFrame(train_samples)
    print(f"Created train_df with {len(train_df)} samples.")
    print("Columns in train_df:")
    print(train_df.columns)
    unique_user_ids = behaviors_df['UserID'].unique()
    user_category_profiles = user_category_profiles.loc[unique_user_ids]

    # Check for any missing users
    missing_user_ids = set(unique_user_ids) - set(user_category_profiles.index)
    if missing_user_ids:
        print(f"Warning: {len(missing_user_ids)} 'UserID's are missing from user_category_profiles.")
        behaviors_df = behaviors_df[~behaviors_df['UserID'].isin(missing_user_ids)]
    else:
        print("All 'UserID's are present in user_category_profiles.")

    # Perform clustering on user_category_profiles
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    user_clusters = kmeans.fit_predict(user_category_profiles)
    print(f"Assigned clusters to users. Number of clusters: {num_clusters}")
    user_cluster_df = pd.DataFrame({
        'UserID': user_category_profiles.index,
        'Cluster': user_clusters
    })
    print("Assigning cluster labels to train_df using map...")
    user_cluster_mapping = dict(zip(user_cluster_df['UserID'], user_cluster_df['Cluster']))
    train_df['Cluster'] = train_df['UserID'].map(user_cluster_mapping)

    missing_clusters = train_df[train_df['Cluster'].isna()]
    if not missing_clusters.empty:
        print(f"{len(missing_clusters)} samples have missing cluster assignments.")
        train_df = train_df.dropna(subset=['Cluster'])
    else:
        print("All samples have cluster assignments.")

    train_df['Cluster'] = train_df['Cluster'].astype(int)
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} test samples.")
    min_cluster_size = train_df['Cluster'].value_counts().min()

    balanced_data = []
    # Iterate over each cluster and sample data to balance
    for cluster in train_df['Cluster'].unique():
        cluster_data = train_df[train_df['Cluster'] == cluster]
        balanced_cluster_data = cluster_data.sample(n=min_cluster_size, random_state=42)
        balanced_data.append(balanced_cluster_data)

    # Combine balanced data for all clusters
    balanced_train_df = pd.concat(balanced_data)
    print("\nlabel balance (0 vs 1):")
    print(train_df['Label'].value_counts())
    if downsampling:
        # Update train_df with the balanced data
        # Count how many 0s and 1s
        label_counts = balanced_train_df['Label'].value_counts()
        min_label_count = label_counts.min()

        balanced_labels = []
        for label_value in balanced_train_df['Label'].unique():
            label_data = balanced_train_df[balanced_train_df['Label'] == label_value]
            # Downsample to the min_label_count to balance the label distribution
            balanced_label_data = label_data.sample(n=min_label_count, random_state=42)
            balanced_labels.append(balanced_label_data)

        # Combine the label balanced data
        final_balanced_train_df = pd.concat(balanced_labels, ignore_index=True)

        # Shuffle the final dataset to mix up the rows
        final_balanced_train_df = final_balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print("\nAfter label balancing (0 vs 1):")
        print(final_balanced_train_df['Label'].value_counts())

        # Now final_balanced_train_df is balanced both by cluster and by label
        train_df = final_balanced_train_df

    #train_df = balanced_train_df.reset_index(drop=True)
    print("Balanced cluster sizes:")
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} samples")
    print("Balanced dataset:")
    print(train_df['Cluster'].value_counts())
    """
    clustered_data_balanced = {}
    min_cluster_size = float('inf')
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} test samples.")
        min_cluster_size = len(cluster_data) if len(cluster_data) < min_cluster_size else min_cluster_size

    for cluster in range(num_clusters):
        data = train_df[train_df['Cluster'] == cluster]
        if len(data) > min_cluster_size:
            clustered_data_balanced[cluster] = data.sample(n=min_cluster_size, random_state=42)
        else:
            clustered_data_balanced[cluster] = data

    print("Balanced cluster sizes:")
    for cluster, data in clustered_data_balanced.items():
        print(f"Cluster {cluster}: {len(data)} samples")
    """
    # Random sampling:
    print(f"Original size: {len(train_df)}")
    train_df_sampled = train_df.sample(frac=fraction, random_state=42)
    print(f"Sampled size: {len(train_df_sampled)}")

    train_df = train_df_sampled
    print("Columns in sampled train_df:")
    print(train_df.columns)
    print(f"Cluster:{train_df['Cluster']}")
    print("Splitting data into training and validation sets for each cluster...")
    clustered_data = {}
    clustered_data = make_clustered_data(train_df, num_clusters, test_size=test_size, random_state=42, split_indepently=split_indepently)
    """
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]

        if cluster_data.empty:
            print(f"No data for Cluster {cluster}. Skipping...")
            continue  # Skip to the next cluster
        if test_size > 0.0:
            train_data, val_data = train_test_split(cluster_data, test_size=test_size, random_state=42, stratify=None)
            else:
                train_data = cluster_data
                val_data = {}
        clustered_data[cluster] = {
            'train': train_data.reset_index(drop=True),
            'val': val_data.reset_index(drop=True)
        }
    """
    print(f"Saved after processing: {news_df_pkl}, {train_df_pkl}")
    news_df.to_pickle(news_df_pkl)
    train_df.to_pickle(train_df_pkl)

    print("Columns in behaviors_df:")
    print(behaviors_df.columns)
    print("Columns in train_df:")
    print(train_df.columns)
    #pivots=[25,40,45,49]
    #ks=[10,5,1]
    if process_valid_sets:
        # def make_eval_sets(df, news_text_dict, max_hist=50, max_len=30, pivots=[25,40,45,49], ks=[10,5,1])
        """
        make_eval_sets_vectorized(
            behaviors_df, news_text_dict,
            max_hist=max_history_length,
            max_len=max_title_length,
            pivots=pivots, ks=ks,
            dataset=dataset, dataset_size=dataset_size
        )

        dfs = make_eval_sets(
                behaviors_df, news_text_dict,
                max_hist=max_history_length,
                max_len=max_title_length,
                pivots=pivots, ks=ks,
                dataset=dataset, dataset_size=dataset_size
        )
        dfs = make_eval_sets_light(behaviors_df,
            news_text_dict,
            max_hist=max_history_length,
            max_len=max_title_length,
            pivots=pivots,
            ks=ks,
            dataset=dataset,
            dataset_size=dataset_size,
            frac=eval_frac)
        """

        make_eval_sets_from_train(
                train_df,
                pivots=pivots,
                ks=ks,
                frac=eval_frac,
                dataset=dataset,
                dataset_size=dataset_size
            )

    return clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters

def grouped_update(df, scores, tracker, uid_col="UserID"):
    """
    Feed one DataFrame batch and its model scores into tracker.

    Parameters
    ----------
    df      : DataFrame with 'Label' and uid_col columns
    scores  : 1‑D NumPy array, aligned with df rows
    tracker : RankingMetricsTracker instance
    """
    df = df.copy()
    df["Score"] = scores
    for _, grp in df.groupby(uid_col):
        tracker.update(grp["Label"].values, grp["Score"].values)

def run_eval(df, model, title_tensor, id2index,
             batch_size=512, k_vals=(5, 10, 20)):
    """
    Evaluate *model* on *df* using RankingMetricsTracker.

    df must contain ['UserID', 'CandidateID', 'Label', 'HistoryTitles'].
    """
    tracker = RankingMetricsTracker(k_values=k_vals)

    # simple batching by row index
    for _, batch in df.groupby(np.arange(len(df)) // batch_size, sort=False):
        hist  = tf.stack(batch["HistoryTitles"].to_list())            # [B,50,30]
        #indices = [id2index[nid] for nid in batch["CandidateID"]]
        pad_idx = id2index.setdefault("_PAD_", title_tensor.shape[0])   # one-time
        indices = [id2index.get(nid, pad_idx) for nid in batch["CandidateID"]]

        cand    = tf.gather(title_tensor, indices)
        hist  = tf.convert_to_tensor(hist,  dtype=tf.int32)
        cand  = tf.convert_to_tensor(cand,  dtype=tf.int32)
        scores = model([hist, cand]).numpy()
        #scores = model(hist, cand).numpy()                            # [B]

        grouped_update(batch, scores, tracker)   # <─ one call per batch

    return tracker.result()

from tqdm import tqdm

def explode_slate(df):
    tmp = df["Impressions"].str.split(expand=True).stack()
    out = df.loc[tmp.index.get_level_values(0)].copy()
    pair = tmp.str.split('-', n=1, expand=True)
    out["CandidateID"] = pair[0].values
    out["Label"]       = pair[1].astype("int8").values
    return out.reset_index(drop=True)


import dask.dataframe as dd
from dask.diagnostics import ProgressBar

def explode_slate_dask(df, npartitions=28):
    ddf = dd.from_pandas(df, npartitions=npartitions)
    ddf2 = (
        ddf
        .assign(Impression=ddf["Impressions"].str.split(" "))
        .explode("Impression")
    )
    pair = ddf2["Impression"].str.split("-", n=1, expand=True)
    ddf2["CandidateID"] = pair[0]
    ddf2["Label"]       = pair[1].astype("int8")

    with ProgressBar():
        result = ddf2.drop(columns="Impression").compute()

    return result
def cache_titles(text_dict, max_len):
    ids, mats = zip(*text_dict.items())  # mats is a sequence of numpy arrays

    # build a list of Python lists of length exactly max_len
    padded = []
    for mat in mats:
        # ensure it's a Python list
        seq = list(mat)
        if len(seq) >= max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [0] * (max_len - len(seq))
        padded.append(seq)

    # now stack into a NumPy array
    X = np.array(padded, dtype=np.int32)

    # map each NewsID to its row in X
    id2row = {nid: idx for idx, nid in enumerate(ids)}
    return id2row, X

def pad_tensor(hist_ids, id2row, tbl, max_hist, pad_row):
    out = np.empty((max_hist, tbl.shape[1]), dtype="int32")
    out[:] = pad_row                                    # pre-fill zeros
    idx = np.fromiter((id2row.get(nid, -1) for nid in hist_ids[-max_hist:]),
                      dtype="int32")
    if len(idx):
        out[-len(idx):] = tbl[idx]          # gather rows in one shot
    return out

import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from numba.typed import List
from typing import Dict, Any

def make_eval_sets_vectorized(
    df: pd.DataFrame,
    news_text_dict: Dict[int,str],
    max_hist: int = 50,
    max_len: int = 30,
    pivots: list[int] = [25,40,45,49],
    ks: list[int]    = [10,5,1],
    dataset: str     = "valid",
    dataset_size: str= "small"
) -> None:
    """
    Builds your three (full / last-k / swap-pivot) evaluation sets in a single
    fast, vectorized pass.  Saves each to
      evaluation_{dataset}_{dataset_size}_{variant}.parquet
    """
    print("make_eval_sets_vectorized")
    # --- 1) PREPROCESS & EXPLODE ------------------------------------------------

    df = df.copy()
    # parse Time if necessary
    if df["Time"].dtype == object:
        df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")

    print("explode_slate_dask")
    # explode your slate / build raw token-ID lists
    base = explode_slate_dask(df)  
    # assume HistoryText is a space-separated string of integer token IDs
    print("process history")
    base["HistIDs"] = base["HistoryText"] \
                         .fillna("") \
                         .str.split() \
                         .map(lambda toks: [int(t[1:]) for t in toks])

    print("build pools")
    # build daily donor pools (as before)
    base["Date"] = base["Time"].dt.floor("D")
    pools = (base.groupby("Date")["HistIDs"]
                 .agg(lambda s: list({tuple(h) for h in s if h}))
                 .to_dict())

    rng = np.random.default_rng(0)
    def pick_donor(hist: list[int], date) -> list[int]:
        pool = pools[date]
        if len(pool) == 1:
            return hist[-max_hist:]
        donor = pool[rng.integers(len(pool))]
        return donor[-max_hist:]

    base["DonorHist"] = [
        pick_donor(h, d) for h,d in zip(base["HistIDs"], base["Date"])
    ]

    print("conv to list")
    # turn your two columns into Python lists (once)
    hist_lists  = base["HistIDs"].tolist()
    donor_lists = base["DonorHist"].tolist()
    N = len(hist_lists)

    # --- 2) CACHE TOKENS --------------------------------------------------------

    # this is identical to your cache_titles call
    print("cache_titles")
    id2row, title_tbl = cache_titles(news_text_dict, max_len)

    # build a dense numpy lookup for id2row[tok]
    int_keys = [int(tok) for tok in id2row.keys()]
    max_tok  = max(int_keys)
    print(f"max token ID = {max_tok}")

    id2row_arr = np.full((max_tok + 1,), -1, dtype=np.int32)
    for tok_str, rid in id2row.items():
        tok = int(tok_str)
        id2row_arr[tok] = rid
    
    print("id2row_arr")
    # convert your Python lists into Numba‐typed lists …
    nb_hist  = List()
    nb_donor = List()
    for h,d in zip(hist_lists, donor_lists):
        nb_hist.append(List(h))
        nb_donor.append(List(d))

    print("njit")
    # JIT‐compile the array builder …
    @njit
    def build_array(all_hists, id2row_arr, max_hist, max_len):
        n = len(all_hists)
        out = np.zeros((n, max_hist, max_len), dtype=np.int32)
        for i in range(n):
            hist = all_hists[i]
            L    = len(hist)
            start = max(0, L - max_hist)
            for j in range(start, L):
                tokens = hist[j]
                row    = j - start
                tmax   = min(len(tokens), max_len)
                for t in range(tmax):
                    out[i, row, t] = id2row_arr[tokens[t]]
        return out

    # full histories
    print(" ▸ JIT‐building full-history tensor…")
    hist_full = build_array(nb_hist, id2row_arr, max_hist, max_len)

    # donor histories (for swap)
    print(" ▸ JIT‐building donor‐history tensor…")
    hist_donor = build_array(nb_donor, id2row_arr, max_hist, max_len)


    # --- 4) SLICE OUT YOUR VARIANTS ----------------------------------------------

    # “last-k” variants
    hist_k = {k: hist_full[:, -k:, :] for k in ks}

    # “pivot-swap” variants
    hist_swap = {}
    for p in pivots:
        # take last‐p rows from donor, then last‐(max_hist-p) from full
        first  = hist_donor[:, -p:, :]
        second = hist_full[:, -(max_hist-p):, :]
        hist_swap[p] = np.concatenate([first, second], axis=1)


    # --- 5) DUMP TO PARQUET ------------------------------------------------------

    keep = ["UserID", "CandidateID", "Label"]

    # helper to wrap a (N,*,*) array as a list‐column + write
    def df_and_write(arr: np.ndarray, name: str):
        df_out = base[keep].copy()
        # store the tensor as a Python object per‐row
        df_out["HistoryTitles"] = list(arr)
        # now add your candidate token features
        df_tok = add_candidate_tokens(df_out, news_text_dict)
        path = f"evaluation_{dataset}_{dataset_size}_{name}.parquet"
        df_tok.to_parquet(path, index=False)
        print("   → wrote", path)

    # full
    print("▸ Writing full history…")
    df_and_write(hist_full, "full")

    # last-k
    for k, arr in hist_k.items():
        print(f"▸ Writing last-{k} history…")
        df_and_write(arr, f"k{k}")

    # pivot-swap
    for p, arr in hist_swap.items():
        print(f"▸ Writing swap-{p} history…")
        df_and_write(arr, f"swap{p}")

    print("✅ Done.")



def make_eval_sets(df, news_text_dict, max_hist=50, max_len=30, pivots=[25,40,45,49], ks=[10,5,1], dataset="valid", dataset_size="small"):
    print("make_eval_sets")

    df = df.copy()
    if df["Time"].dtype == "O":
        df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")

    base = explode_slate(df)
    base["HistIDs"] = base["HistoryText"].fillna("").str.split()
    print("base")

    # ---- build donor pools once per day (pure C) ------------------------
    base["Date"] = base["Time"].dt.floor("D")          # 7 unique dates
    pools = (base.groupby('Date')["HistIDs"]
                  .agg(lambda s: list({tuple(h) for h in s.dropna()}))
                  .to_dict())
    print("pools done")

    rng = np.random.default_rng(0)
    def pick_donor(row):
        pool = pools[row.Date]
        if len(pool) == 1:
            return row.HistIDs[-max_hist:]
        donor = pool[rng.integers(len(pool))]
        return donor[-max_hist:]

    base["DonorHist"] = base.apply(pick_donor, axis=1)
    print("donors added")
    cols = ["Hist_full"]

    # ---- three history variants (no Python loops) -----------------------
    for pivot in pivots:
        cols.append(f"Hist_swap{pivot}")
        base[f"Hist_swap{pivot}"] = base.apply(
            lambda r: list(r.DonorHist)[-pivot:]
                    + list(r.HistIDs)[-(max_hist - pivot):],
            axis=1
        )
    for k in ks:
        cols.append(f"Hist_k{k}")
        base[f"Hist_k{k}"] = base["HistIDs"].str[-k:]
    #base["Hist_swap"]  = (base["DonorHist"].str[-pivot:]
    #                      + base["HistIDs"].str[-(max_hist-pivot):])
    base["Hist_full"]  = base["HistIDs"]
    print("hist variants done")

    # ---- build tensors --------------------------------------------------
    id2row, title_tbl  = cache_titles(news_text_dict, max_len)
    pad_row = np.zeros((max_len,), dtype="int32")

    print("building hist tensors")
    tqdm.pandas(desc="tensor")
    #for col in ("Hist_full", "Hist_k10", "Hist_swap"):
    for col in cols:
        base[col] = base[col].progress_apply(
            pad_tensor,
            args=(id2row, title_tbl, max_hist, pad_row)
        )
    print("finishing")

    keep = ["UserID", "CandidateID", "Label"]
    dfs = []
    for col in cols:
        log_print(f"finishing processing {col}")
        df = base[keep+[col] ].rename(columns={col:"HistoryTitles"})
        df = df.reset_index(drop=True)
        df = add_candidate_tokens(df, news_text_dict)
        df.to_pickle(f"evaluation_{dataset}_{dataset_size}_{col}.pkl")
    
    """df_full = base[keep+["Hist_full"] ].rename(columns={"Hist_full":"HistoryTitles"})
    df_k10  = base[keep+["Hist_k10"]  ].rename(columns={"Hist_k10":"HistoryTitles"})
    df_swap = base[keep+["Hist_swap"] ].rename(columns={"Hist_swap":"HistoryTitles"})
    return (df_full.reset_index(drop=True),
            df_k10.reset_index(drop=True),
            df_swap.reset_index(drop=True))
    """
    return dfs

from functools import partial
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm

def _slice_history(hist, keep):
    """Return the *last* `keep` items from a list; used vectorised."""
    return hist[-keep:] if len(hist) > keep else hist

def sample_behaviors(df: pd.DataFrame, frac: float, seed: int = 42):
    #Return a random subset of impressions (not rows)
    keep_impr = (
        df["ImpressionID"]
        .drop_duplicates()
        .sample(frac=frac, random_state=seed)
    )
    return df[df["ImpressionID"].isin(keep_impr)]

def _to_list_int32(x):
    """Arrow accepts nested Python lists, not 2-D ndarrays."""
    # 1. Arrow scalars ─> Python → list
    if hasattr(x, "as_py"):         # ArrowExtensionArray element
        x = x.as_py()               # list / nested list

    # 2. NumPy arrays → list
    if isinstance(x, np.ndarray):
        return x.astype(np.int32).tolist()

    # 3. Python list → enforce int32 dtype element-wise
    return np.asarray(x, dtype=np.int32).tolist()

def process_and_to_parquet(out: pd.DataFrame, path: str) -> None:
    for col in ("HistoryTitles", "CandidateTitleTokens"):
        out[col] = [_to_list_int32(v) for v in out[col]._values]

    # now every cell is a *list* (1-D for Candidate, 2-D for History)
    out.to_parquet(path, engine="pyarrow", row_group_size=100_000)

def make_eval_sets_light(df,
                          news_text_dict,
                          max_hist=50,
                          max_len=30,
                          pivots=(45,),
                          ks=(5,),
                          dataset="valid",
                          dataset_size="small",
                          rng_seed=0,
                          out_dir=".",
                          compression="snappy",
                          frac=1.0):
    """
    Build compact evaluation files that differ only in the HistoryTitles
    column.  Every other column (UserID, CandidateID, Label,
    CandidateTitleTokens) is identical to train_df.
    """
    print(f"frac:{frac}")
    df = df.copy()
    if frac < 1.0:
        subset_tag = f"_{frac}"
        df = sample_behaviors(df, frac)
    else:
        subset_tag = ""
    if df["Time"].dtype == "O":
        df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")

    base = explode_slate(df)                       # identical to training
    base["HistIDs"] = base["HistoryText"].fillna("").str.split()

    base["CandidateTitleTokens"] = base["CandidateID"].map(
        lambda nid: news_text_dict.get(nid, [0]*max_len)
    )

    # ---------------- donor pools – one list per day --------------------
    base["Date"] = base["Time"].dt.floor("D")
    pools = (base.groupby("Date")["HistIDs"]
                   .agg(lambda s: list({tuple(h) for h in s if h}))
                   .to_dict())
    log_print(f"pools:{pools}")

    rng = np.random.default_rng(rng_seed)
    def _pick_donor(hist_ids, date):
        pool = pools[date]
        if len(pool) <= 1:
            return hist_ids
        donor_tuple = pool[rng.integers(len(pool))]
        return list(donor_tuple)
    def _pick_donor2(hist_ids, date):
        pool = pools[date]
        donor = rng.choice(pool) if len(pool) > 1 else hist_ids
        return donor

    base["DonorHist"] = [ _pick_donor(h,d) for h,d in
                          zip(base["HistIDs"], base["Date"]) ]

    # ----------------- write one Parquet per variant --------------------
    def encode_history(hist_ids):
        """Turn ['N123', 'N456', …] → list[list[int]] length max_hist."""
        rows = []
        for nid in hist_ids[-max_hist:]:
            rows.append(news_text_dict.get(nid, [0]*max_len)[:max_len])
        pad = max_hist - len(rows)
        if pad:
            rows = [[0]*max_len]*pad + rows
        return rows

    base["HistoryTitles"] = base["HistIDs"].apply(encode_history)
    keep_cols = ["UserID", "CandidateID", "Label",
                 "CandidateTitleTokens", "HistoryTitles"]

    def _fname(stem):
        return (f"{out_dir}/evaluation_{dataset}_{dataset_size}"
                f"{subset_tag}_{stem}.parquet")

    # full-history (unchanged)
    #out = base[keep_cols + ["HistIDs"]].rename(
    #        columns={"HistIDs": "HistoryTitles"})
    out = base[keep_cols].copy()
    #out["HistoryTitles"]        = out["HistoryTitles"].astype("object")
    #out["CandidateTitleTokens"] = out["CandidateTitleTokens"].astype("object")
    #print(f"processing HistoryTitles and CandidateTitleTokens")
    #for col in ("HistoryTitles", "CandidateTitleTokens"):
    #    out[col] = out[col].to_pylist().apply(lambda x: np.asarray(x, dtype="int32"))
    #print(f"finshed processing HistoryTitles and CandidateTitleTokens")
    #path = _fname("Hist_full")
    #out.to_parquet(path, engine="pyarrow", compression=compression, index=False)
    #out.to_parquet(path, engine="pyarrow", row_group_size=100_000)
    path = _fname("Hist_full")
    process_and_to_parquet(out, path)

    # last-k variants
    for k in ks:
        out = base.copy()
        out["HistoryTitles"] = out["HistIDs"].map(partial(_slice_history,
                                                          keep=k))
        #out["HistoryTitles"]        = out["HistoryTitles"].astype("object")
        #out["CandidateTitleTokens"] = out["CandidateTitleTokens"].astype("object")
        path = _fname(f"Hist_k{k}")
        process_and_to_parquet(out[keep_cols], path)
        """
        out[keep_cols].to_parquet(
            _fname(f"Hist_k{k}"),
            engine="pyarrow",
            row_group_size=100_000,
            compression="zstd")
        """


    # donor-swap variants
    for p in pivots:
        def _swap(row, pivot=p):
            donor_part = row.DonorHist[-pivot:]
            own_part   = row.HistIDs[-(max_hist - pivot):]
            return donor_part + own_part
        out = base.copy()
        out["HistoryTitles"] = out.apply(_swap, axis=1)
        path = _fname(f"Hist_swap{p}")
        process_and_to_parquet(out[keep_cols], path)
        #out["HistoryTitles"]        = out["HistoryTitles"].astype("object")
        #out["CandidateTitleTokens"] = out["CandidateTitleTokens"].astype("object")
        """
        out[keep_cols].to_parquet(
            _fname(f"Hist_swap{p}"),
            engine="pyarrow",
            row_group_size=100_000,
            compression="zstd")
        """

class DataGenerator(Sequence):
    def __init__(self, df, batch_size, max_history_length=50, max_title_length=30):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.max_history_length = max_history_length
        self.max_title_length = max_title_length
        self.indices = np.arange(len(self.df))
        #print(f"[DataGenerator] Initialized with {len(self.df)} samples and batch_size={self.batch_size}")

    def __len__(self):
        length = int(np.ceil(len(self.df) / self.batch_size))
        #print(f"[DataGenerator] Number of batches per epoch: {length}")
        return length

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.df))
        batch_indices = self.indices[start:end]
        batch_df = self.df.iloc[batch_indices]

        #print(f"[DataGenerator] Generating batch {idx+1}/{self.__len__()} with samples {start} to {end}")

        if len(batch_df) == 0:
            print(f"[DataGenerator] Warning: Batch {idx} is empty.")
            return None, None

        # Initialize batches
        history_batch = []
        candidate_batch = []
        labels_batch = []

        for _, row in batch_df.iterrows():
            # Get tokenized history titles
            history_titles = row['HistoryTitles']  # List of lists of integers

            # Pad each title in history
            history_titles_padded = pad_sequences(
                history_titles,
                maxlen=self.max_title_length,
                padding='post',
                truncating='post',
                value=0
            )

            # Pad or truncate the history to MAX_HISTORY_LENGTH
            if len(history_titles_padded) < self.max_history_length:
                padding = np.zeros((self.max_history_length - len(history_titles_padded), self.max_title_length), dtype='int32')
                history_titles_padded = np.vstack([padding, history_titles_padded])
            else:
                history_titles_padded = history_titles_padded[-self.max_history_length:]

            # Get candidate title tokens
            candidate_title = row['CandidateTitleTokens']  # List of integers
            candidate_title_padded = pad_sequences(
                [candidate_title],
                maxlen=self.max_title_length,
                padding='post',
                truncating='post',
                value=0
            )[0]

            # Append to batches
            history_batch.append(history_titles_padded)
            candidate_batch.append(candidate_title_padded)
            labels_batch.append(row['Label'])

        history_batch = np.array(history_batch, dtype='int32')  # Shape: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH)
        candidate_batch = np.array(candidate_batch, dtype='int32')  # Shape: (batch_size, MAX_TITLE_LENGTH)
        labels_batch = np.array(labels_batch, dtype='float32')  # Shape: (batch_size,)
        inputs = {
            'history_input': history_batch,
            'candidate_input': candidate_batch
        }

        return inputs, labels_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

@register_keras_serializable()
class SqueezeLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super(SqueezeLayer, self).get_config()
        config.update({'axis': self.axis})
        return config

@register_keras_serializable()
class ExpandDimsLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({'axis': self.axis})
        return config

@register_keras_serializable()
class SumPooling(Layer):
    def __init__(self, axis=1, **kwargs):
        super(SumPooling, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def get_config(self):
        config = super(SumPooling, self).get_config()
        config.update({'axis': self.axis})
        return config

@register_keras_serializable()
class Fastformer(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        super(Fastformer, self).__init__(**kwargs)
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head

        self.WQ = None
        self.WK = None
        self.WV = None
        self.WO = None

    def build(self, input_shape):
        self.WQ = Dense(self.output_dim, use_bias=False, name='WQ')
        self.WK = Dense(self.output_dim, use_bias=False, name='WK')
        self.WV = Dense(self.output_dim, use_bias=False, name='WV')
        self.WO = Dense(self.output_dim, use_bias=False, name='WO')
        super(Fastformer, self).build(input_shape)

    def call(self, inputs):
        if len(inputs) == 2:
            Q_seq, K_seq = inputs
            Q_mask = None
            K_mask = None
        elif len(inputs) == 4:
            Q_seq, K_seq, Q_mask, K_mask = inputs

        batch_size = tf.shape(Q_seq)[0]
        seq_len = tf.shape(Q_seq)[1]

        # Linear projections
        Q = self.WQ(Q_seq)  # Shape: (batch_size, seq_len, output_dim)
        K = self.WK(K_seq)  # Shape: (batch_size, seq_len, output_dim)
        V = self.WV(K_seq)  # Shape: (batch_size, seq_len, output_dim)

        # Reshape for multi-head attention
        Q = tf.reshape(Q, (batch_size, seq_len, self.nb_head, self.size_per_head))
        K = tf.reshape(K, (batch_size, seq_len, self.nb_head, self.size_per_head))
        V = tf.reshape(V, (batch_size, seq_len, self.nb_head, self.size_per_head))

        # Compute global query and key
        global_q = tf.reduce_mean(Q, axis=1, keepdims=True)  # (batch_size, 1, nb_head, size_per_head)
        global_k = tf.reduce_mean(K, axis=1, keepdims=True)  # (batch_size, 1, nb_head, size_per_head)

        # Compute attention weights
        weights = global_q * K + global_k * Q  # (batch_size, seq_len, nb_head, size_per_head)
        weights = tf.reduce_sum(weights, axis=-1)  # (batch_size, seq_len, nb_head)
        weights = tf.nn.softmax(weights, axis=1)  # Softmax over seq_len

        # Apply attention weights to values
        weights = tf.expand_dims(weights, axis=-1)  # (batch_size, seq_len, nb_head, 1)
        context = weights * V  # (batch_size, seq_len, nb_head, size_per_head)

        # Combine heads
        context = tf.reshape(context, (batch_size, seq_len, self.output_dim))

        # Final projection
        output = self.WO(context)  # (batch_size, seq_len, output_dim)

        return output  # Output shape: (batch_size, seq_len, output_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super(Fastformer, self).get_config()
        config.update({
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head
        })
        return config

@register_keras_serializable()
class NewsEncoder(Layer):
    def __init__(self, vocab_size, embedding_dim=256, dropout_rate=0.2, nb_head=8, size_per_head=32, embedding_layer=None, **kwargs):
        super(NewsEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.nb_head = nb_head
        self.size_per_head = size_per_head

        # Define sub-layers
        self.embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name='embedding_layer'
        )
        self.dropout = Dropout(self.dropout_rate)
        self.dense = Dense(1)
        self.softmax = Softmax(axis=1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

        self.fastformer_layer = Fastformer(nb_head=self.nb_head, size_per_head=self.size_per_head, name='fastformer_layer')

    def build(self, input_shape):
        super(NewsEncoder, self).build(input_shape)

    def call(self, inputs):
        # Create mask
        mask = tf.cast(tf.not_equal(inputs, 0), dtype='float32')  # Shape: (batch_size, seq_len)

        # Embedding
        title_emb = self.embedding_layer(inputs)  # Shape: (batch_size, seq_len, embedding_dim)
        title_emb = self.dropout(title_emb)

        # Fastformer
        hidden_emb = self.fastformer_layer([title_emb, title_emb, mask, mask])  # Shape: (batch_size, seq_len, embedding_dim)
        hidden_emb = self.dropout(hidden_emb)

        # Attention-based Pooling
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, seq_len, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, seq_len, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, seq_len, embedding_dim)
        news_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)

        return news_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(NewsEncoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head,
            'embedding_layer': tf.keras.utils.serialize_keras_object(self.embedding_layer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_layer_config = config.pop('embedding_layer', None)
        embedding_layer = tf.keras.layers.deserialize(embedding_layer_config) if embedding_layer_config else None
        return cls(embedding_layer=embedding_layer, **config)

@register_keras_serializable()
class MaskLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mask = tf.cast(tf.not_equal(inputs, 0), dtype='float32')
        return mask

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        return config

@register_keras_serializable()
class UserEncoder(Layer):
    def __init__(self, news_encoder_layer, embedding_dim=256, **kwargs):
        super(UserEncoder, self).__init__(**kwargs)
        self.news_encoder_layer = news_encoder_layer
        self.embedding_dim = embedding_dim
        self.dropout = Dropout(0.2)
        self.layer_norm = LayerNormalization()
        self.fastformer = Fastformer(nb_head=8, size_per_head=32, name='user_fastformer')
        self.dense = Dense(1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.softmax = Softmax(axis=1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

    def call(self, inputs):
        # inputs: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH)
        # Encode each news article in the history
        news_vectors = TimeDistributed(self.news_encoder_layer)(inputs)  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)

        # Step 1: Create a boolean mask
        mask = tf.not_equal(inputs, 0)  # Shape: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH), dtype=bool

        # Step 2: Reduce along the last axis
        mask = tf.reduce_any(mask, axis=-1)  # Shape: (batch_size, MAX_HISTORY_LENGTH), dtype=bool

        # Step 3: Cast to float32 if needed
        mask = tf.cast(mask, dtype='float32')  # Shape: (batch_size, MAX_HISTORY_LENGTH), dtype=float32

        # Fastformer
        hidden_emb = self.fastformer([news_vectors, news_vectors, mask, mask])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        hidden_emb = self.dropout(hidden_emb)
        hidden_emb = self.layer_norm(hidden_emb)

        # Attention-based Pooling over history
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        user_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)

        return user_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(UserEncoder, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'news_encoder_layer': tf.keras.utils.serialize_keras_object(self.news_encoder_layer),
        })
        return config
    @classmethod
    def from_config(cls, config):
        # Extract the serialized news_encoder_layer config
        news_encoder_config = config.pop("news_encoder_layer")
        # Reconstruct the news_encoder_layer instance
        news_encoder_layer = tf.keras.utils.deserialize_keras_object(
            news_encoder_config, custom_objects={'NewsEncoder': NewsEncoder}
        )
        return cls(news_encoder_layer, **config)

def build_model(vocab_size, max_title_length=30, max_history_length=50, embedding_dim=256, nb_head=8, size_per_head=32, dropout_rate=0.2, timed=False):
    # Define Inputs
    history_input = Input(shape=(max_history_length, max_title_length), dtype='int32', name='history_input')
    candidate_input = Input(shape=(max_title_length,), dtype='int32', name='candidate_input')

    # Instantiate NewsEncoder Layer
    if timed:
        news_encoder_layer = TimedNewsEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            nb_head=nb_head,
            size_per_head=size_per_head,
            name='news_encoder'
        )
    else:
        news_encoder_layer = NewsEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            nb_head=nb_head,
            size_per_head=size_per_head,
            name='news_encoder'
        )
    # Encode Candidate News
    candidate_vector = news_encoder_layer(candidate_input)  # Shape: (batch_size, embedding_dim)

    # Encode User History
    if timed:
        user_vector = TimedUserEncoder(news_encoder_layer, embedding_dim=embedding_dim, name='timed_user_encoder')(history_input)  # Shape: (batch_size, embedding_dim)
    else:
        user_vector = UserEncoder(news_encoder_layer, embedding_dim=embedding_dim, name='user_encoder')(history_input)  # Shape: (batch_size, embedding_dim)

    # Scoring Function: Dot Product between User and Candidate Vectors
    score = Dot(axes=-1)([user_vector, candidate_vector])  # Shape: (batch_size, 1)
    score = Activation('sigmoid')(score)  # Shape: (batch_size, 1)

    # Build Model
    model = Model(inputs={'history_input': history_input, 'candidate_input': candidate_input}, outputs=score)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='AUC')
        ]
    )

    return model

# ----------------------------
# NRMS‐style NewsEncoder
# ----------------------------
@register_keras_serializable()
class NewsEncoderNRMS(Layer):
    def __init__(self,
                 vocab_size,
                 embedding_dim=256,
                 num_heads=8,
                 dropout_rate=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dropout = Dropout(dropout_rate)
        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim//num_heads,
                                      dropout=dropout_rate)
        self.dense = Dense(1)
        self.softmax = Softmax(axis=1)
        self.squeeze = SqueezeLayer(-1)
        self.expand = ExpandDimsLayer(-1)
        self.sum_pool = SumPooling(1)

    def call(self, x):
        # x: [B, L]
        mask = tf.cast(tf.not_equal(x, 0), tf.bool)            # [B, L]
        emb = self.embed(x)                                    # [B, L, D]
        emb = self.dropout(emb)
        # self‐attention (query=key=emb)
        attn_out = self.mha(query=emb, value=emb, key=emb,
                            attention_mask=mask[:, tf.newaxis, :])  # [B, L, D]
        attn_out = self.dropout(attn_out)
        # attention pooling
        scores = self.dense(attn_out)     # [B, L, 1]
        scores = self.squeeze(scores)     # [B, L]
        weights = self.softmax(scores)    # [B, L]
        weights = self.expand(weights)    # [B, L, 1]
        weighted = attn_out * weights     # [B, L, D]
        return self.sum_pool(weighted)    # [B, D]

# ----------------------------
# NRMS‐style UserEncoder
# ----------------------------
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dense
import tensorflow as tf

class oldUserEncoderNRMS(Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Create your MHA, layernorm, etc in __init__
        self.mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            name="user_mha"
        )
        self.layer_norm = LayerNormalization(name="user_ln")
        self.dense = Dense(self.embed_dim, activation="tanh", name="user_dense")

    def build(self, input_shape):
        # input_shape will be (batch_size, history_len, embed_dim)
        # by the time you call UserEncoderNRMS, news_encoder should already have
        # turned your raw token IDs into embed vectors of size `embed_dim`.
        # So here we just mark this layer as built.
        super().build(input_shape)
        # (No need to manually call self.mha.build — Keras will do it for you now
        # that you've implemented build())

    def call(self, history_embeddings):
        # history_embeddings: Tensor [B, H, D]
        # we'll do self-attention over the history:
        attn_output = self.mha(
            query=history_embeddings,
            key=history_embeddings,
            value=history_embeddings
        )  # -> [B, H, D]
        attn_output = self.layer_norm(attn_output + history_embeddings)
        # pool across the H dimension to get a single user vector [B, D]
        user_vec = tf.reduce_mean(attn_output, axis=1)
        user_vec = self.dense(user_vec)
        return user_vec  # shape [B, D]

    def compute_output_shape(self, input_shape):
        # we're collapsing the H dimension into the embedding dim
        batch_size = input_shape[0]
        return (batch_size, self.embed_dim)


@register_keras_serializable()
class oldUserEncoderNRMS(Layer):
    def __init__(self,
                 news_encoder: Layer,
                 embedding_dim=256,
                 num_heads=8,
                 dropout_rate=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embedding_dim
        self.num_heads = num_heads
        # Create your MHA, layernorm, etc in __init__
        #self.mha = MultiHeadAttention(
        #    num_heads=self.num_heads,
        #    key_dim=self.embed_dim // self.num_heads,
        #    name="user_mha"
        #)
        self.layer_norm = LayerNormalization(name="user_ln")
        #self.dense = Dense(self.embed_dim, activation="tanh", name="user_dense")
        self.news_enc = news_encoder
        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim//num_heads,
                                      dropout=dropout_rate)
        self.dropout = Dropout(dropout_rate)
        self.dense = Dense(1)
        self.softmax = Softmax(axis=1)
        self.squeeze = SqueezeLayer(-1)
        self.expand = ExpandDimsLayer(-1)
        self.sum_pool = SumPooling(1)

    def build(self, input_shape):
        # input_shape will be (batch_size, history_len, embed_dim)
        # by the time you call UserEncoderNRMS, news_encoder should already have
        # turned your raw token IDs into embed vectors of size `embed_dim`.
        # So here we just mark this layer as built.
        super().build(input_shape)
        # (No need to manually call self.mha.build — Keras will do it for you now
        # that you've implemented build())
    def call(self, history_inputs):
        # history_inputs: [B, H, L]
        B, H, L = tf.shape(history_inputs)[0], tf.shape(history_inputs)[1], tf.shape(history_inputs)[2]
        flat = tf.reshape(history_inputs, (-1, L))             # [B*H, L]
        # encode each news in history
        news_vecs = self.news_enc(flat)                        # [B*H, D]
        news_vecs = tf.reshape(news_vecs, (B, H, -1))         # [B, H, D]
        # build a mask: if a row is all-zero→False
        mask = tf.reduce_any(tf.not_equal(history_inputs, 0), axis=-1)  # [B, H]
        # self-attend across history
        attn = self.mha(query=news_vecs, value=news_vecs, key=news_vecs,
                        attention_mask=mask[:, tf.newaxis, :])       # [B, H, D]
        attn = self.dropout(attn)
        # pooling
        scores = self.dense(attn)     # [B, H, 1]
        scores = self.squeeze(scores) # [B, H]
        weights = self.softmax(scores)# [B, H]
        weights = self.expand(weights)# [B, H, 1]
        weighted = attn * weights     # [B, H, D]
        return self.sum_pool(weighted) # [B, D]
    def compute_output_shape(self, input_shape):
        # we're collapsing the H dimension into the embedding dim
        batch_size = input_shape[0]
        return (batch_size, self.embed_dim)
# ----------------------------
# build a combined NRMS model
# ----------------------------
from tensorflow.keras.optimizers import Adam



import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Softmax, LayerNormalization, MultiHeadAttention
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer

class MaskedGlobalAvgPool(Layer):
    def call(self, inputs, mask=None):
        # inputs: (B, H, D), mask: (B, H) boolean or float
        if mask is None:
            # derive from zero‐padding if you like:
            mask = tf.cast(tf.reduce_any(tf.not_equal(inputs, 0), axis=-1), tf.float32)
        mask = tf.expand_dims(mask, axis=-1)           # (B, H, 1)
        sums = tf.reduce_sum(inputs * mask, axis=1)    # (B, D)
        lengths = tf.reduce_sum(mask, axis=1) + 1e-6    # (B, 1)
        return sums / lengths                          # (B, D)


class NewsEncoderNRMS(Layer):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        max_title_length: int = 30,
        num_heads: int = 8,
        key_dim: int = 32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_title_length = max_title_length
        self.num_heads = num_heads
        self.key_dim = key_dim

        # sub-layers
        self.embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_title_length,
            mask_zero=True,
            name="news_embed"
        )
        # NRMS uses a self-attention over the title tokens:
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            name="news_mha"
        )
        self.layer_norm = LayerNormalization(name="news_ln")
        self.attn_score = Dense(1, name="news_score")
        self.softmax    = Softmax(axis=1, name="news_softmax")

    def call(self, x, **kwargs):
        # inputs: (batch_size, title_len)
        from tensorflow.keras.layers import Lambda

        # instead of:
        #    mask = tf.cast(tf.not_equal(x, 0), tf.float32)
        # do:
        mask = Lambda(
            lambda x: tf.cast(tf.not_equal(x, 0), tf.float32),
            name="mask_hist"
        )(inputs)
        emb = self.embed(x)                                    # [B, L, D]
        emb = self.dropout(emb)
        # self‐attention (query=key=emb)
        attn_out = self.mha(query=emb, value=emb, key=emb,
                            attention_mask=mask[:, tf.newaxis, :])  # [B, L, D]
        # self-attention: query/value/key all = x
        """
        #x = self.embedding(inputs)                              # (B, L, D)
        #q = self.mha(query=x, value=x, key=x)                   # (B, L, D)
        q = self.mha(
            query=x, value=x, key=x,
            attention_mask=mask[:, None, None, :]   # broadcast to (B, heads, T_q, T_k)
        )
        """
        # score each token:
        scores = tf.squeeze(self.attn_score(q), axis=-1)        # (B, L)
        weights = self.softmax(scores)                          # (B, L)
        weights = tf.expand_dims(weights, -1)                   # (B, L, 1)
        # weighted sum → one vector per news
        news_vec = tf.reduce_sum(q * weights, axis=1)           # (B, D)
        return self.layer_norm(news_vec)                        # (B, D)

    def compute_output_shape(self, input_shape):
        # from (batch_size, L) → (batch_size, D)
        return (input_shape[0], self.embedding_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "max_title_length": self.max_title_length,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        })
        return cfg


class oldUserEncoderNRMS(Layer):
    def __init__(
        self,
        news_encoder: NewsEncoderNRMS,
        max_history_length: int = 50,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.news_encoder = news_encoder
        self.max_history_length = max_history_length

        # user-level self-attention over the history vectors
        self.mha_user = MultiHeadAttention(
            num_heads=news_encoder.num_heads,
            key_dim=news_encoder.key_dim,
            name="user_mha"
        )
        self.layer_norm = LayerNormalization(name="user_ln")
        self.attn_score = Dense(1, name="user_score")
        self.softmax    = Softmax(axis=1, name="user_softmax")

    def call(self, inputs, **kwargs):
        # inputs: (batch_size, H, L)
        B, H, L = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        # reshape to run news_encoder on each title separately
        flat = tf.reshape(inputs, (-1, L))             # (B*H, L)
        news_vecs = self.news_encoder(flat)             # (B*H, D)
        news_vecs = tf.reshape(news_vecs, (B, H, -1))   # (B, H, D)

        # user self-attention
        u = self.mha_user(query=news_vecs, value=news_vecs, key=news_vecs)  # (B, H, D)
        scores = tf.squeeze(self.attn_score(u), -1)    # (B, H)
        weights = self.softmax(scores)                 # (B, H)
        weights = tf.expand_dims(weights, -1)          # (B, H, 1)
        user_vec = tf.reduce_sum(u * weights, axis=1)  # (B, D)
        return self.layer_norm(user_vec)               # (B, D)

    def compute_output_shape(self, input_shape):
        # from (B, H, L) → (B, D)
        return (input_shape[0], self.news_encoder.embedding_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "news_encoder": tf.keras.utils.serialize_keras_object(self.news_encoder),
            "max_history_length": self.max_history_length,
        })
        return cfg



import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Embedding, MultiHeadAttention,
    GlobalAveragePooling1D, TimeDistributed,
    Dot, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, register_keras_serializable

# ─── Data generator for NRMS ─────────────────────────────────────────────────────

class DataGeneratorNRMS(Sequence):
    def __init__(self, df, batch_size, max_history_length=50, max_title_length=30):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.max_h = max_history_length
        self.max_t = max_title_length
        self.indices = np.arange(len(self.df))

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = self.df.iloc[batch_idx]

        H, T = self.max_h, self.max_t
        hist_batch = np.zeros((len(batch), H, T), dtype='int32')
        cand_batch = np.zeros((len(batch), T),     dtype='int32')
        y_batch    = np.zeros((len(batch),),        dtype='float32')

        for i, row in enumerate(batch.itertuples()):
            # pad/truncate history
            hist = row.HistoryTitles  # list[list[int]] length ≤ some H_i
            # pad titles
            hist = tf.keras.preprocessing.sequence.pad_sequences(
                hist, maxlen=T, padding='post', truncating='post', value=0)
            # pad history dimension
            if hist.shape[0] < H:
                pad = np.zeros((H - hist.shape[0], T), dtype='int32')
                hist = np.vstack([pad, hist])
            else:
                hist = hist[-H:]
            hist_batch[i] = hist

            # candidate
            cand = row.CandidateTitleTokens
            cand = tf.keras.preprocessing.sequence.pad_sequences(
                [cand], maxlen=T, padding='post', truncating='post', value=0)[0]
            cand_batch[i] = cand

            y_batch[i] = row.Label

        return {'history_input': hist_batch, 'candidate_input': cand_batch}, y_batch


import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, MultiHeadAttention
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable()
class NRMSBlock(Layer):
    def __init__(self, nb_head, size_per_head, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(
            num_heads=nb_head,
            key_dim=size_per_head,
            dropout=dropout_rate,
            name=self.name + "_mha"
        )
        self.dropout = Dropout(dropout_rate, name=self.name + "_dropout")

    def call(self, inputs):
        # unpack
        if len(inputs) == 4:
            Q_seq, K_seq, Q_mask_2d, K_mask_2d = inputs
        else:
            Q_seq, K_seq = inputs
            Q_mask_2d = K_mask_2d = None

        # if they gave you a 2D boolean mask, expand it to 3D:
        #   mask_2d: (batch, seq_len)
        #   ⇒ mask_3d: (batch, seq_len, seq_len)
        if K_mask_2d is not None:
            # make sure it's boolean
            K_mask_2d = tf.cast(K_mask_2d, tf.bool)
            # row‐ and column‐wise AND to get a [batch, q_len, k_len] mask
            mask_3d = tf.logical_and(
                tf.expand_dims(K_mask_2d, axis=1),    # (batch,1,k_len)
                tf.expand_dims(K_mask_2d, axis=2)     # (batch,q_len,1)
            )
        else:
            mask_3d = None

        attn_out = self.mha(
            query=Q_seq,
            value=K_seq,
            key=K_seq,
            attention_mask=mask_3d,    # now shape=(batch, q_len, k_len)
            return_attention_scores=False
        )
        return self.dropout(attn_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "nb_head": self.mha.num_heads,
            "size_per_head": self.mha.key_dim,
            "dropout_rate": self.dropout.rate,
        })
        return cfg


@register_keras_serializable()
class NewsEncoderNRMS(Layer):
    def __init__(self, vocab_size, embedding_dim=256, dropout_rate=0.2, nb_head=8, size_per_head=32, embedding_layer=None, **kwargs):
        super(NewsEncoderNRMS, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.nb_head = nb_head
        self.size_per_head = size_per_head

        # Define sub-layers
        self.embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name='embedding_layer'
        )
        self.dropout = Dropout(self.dropout_rate)
        self.dense = Dense(1)
        self.softmax = Softmax(axis=1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

        #self.fastformer_layer = Fastformer(nb_head=self.nb_head, size_per_head=self.size_per_head, name='fastformer_layer')
        self.nrms_layer = NRMSBlock(
            nb_head=self.nb_head,
            size_per_head=self.size_per_head,
            dropout_rate=self.dropout_rate,
            name="nrms_layer"
        )
    def build(self, input_shape):
        super(NewsEncoderNRMS, self).build(input_shape)

    def call(self, inputs):
        # Create mask
        #mask = tf.cast(tf.not_equal(inputs, 0), dtype='float32')  # Shape: (batch_size, seq_len)
        mask = tf.not_equal(inputs, 0)
        # Embedding
        title_emb = self.embedding_layer(inputs)  # Shape: (batch_size, seq_len, embedding_dim)
        title_emb = self.dropout(title_emb)

        # Fastformer
        #hidden_emb = self.fastformer_layer([title_emb, title_emb, mask, mask])  # Shape: (batch_size, seq_len, embedding_dim)
        #mha_mask = tf.reshape(mask, [tf.shape(mask)[0], 1, 1, tf.shape(mask)[1]])

        # 4) apply your NRMS self‐attention block
        hidden_emb = self.nrms_layer([title_emb, title_emb, mask, mask])
    



        hidden_emb = self.dropout(hidden_emb)

        # Attention-based Pooling
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, seq_len, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, seq_len, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, seq_len, embedding_dim)
        news_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)

        return news_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(NewsEncoderNRMS, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head,
            'embedding_layer': tf.keras.utils.serialize_keras_object(self.embedding_layer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_layer_config = config.pop('embedding_layer', None)
        embedding_layer = tf.keras.layers.deserialize(embedding_layer_config) if embedding_layer_config else None
        return cls(embedding_layer=embedding_layer, **config)

@register_keras_serializable()
class MaskLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mask = tf.cast(tf.not_equal(inputs, 0), dtype='float32')
        return mask

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        return config

@register_keras_serializable()
class UserEncoderNRMS(Layer):
    def __init__(self, news_encoder_layer, embedding_dim=256, **kwargs):
        super(UserEncoderNRMS, self).__init__(**kwargs)
        self.news_encoder_layer = news_encoder_layer
        self.embedding_dim = embedding_dim
        self.dropout = Dropout(0.2)
        self.layer_norm = LayerNormalization()
        #self.fastformer = Fastformer(nb_head=8, size_per_head=32, name='user_fastformer')
        self.nrms_layer = NRMSBlock(
            nb_head=8,
            size_per_head=32,
            name="user_nrms_layer"
        )
        self.dense = Dense(1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.softmax = Softmax(axis=1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

    def call(self, inputs):
        # inputs: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH)
        # Encode each news article in the history
        #news_vectors = TimeDistributed(self.news_encoder_layer)(inputs)  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)

        # Step 1: Create a boolean mask
        #mask = tf.not_equal(inputs, 0)  # Shape: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH), dtype=bool

        # Step 2: Reduce along the last axis
        #mask = tf.reduce_any(mask, axis=-1)  # Shape: (batch_size, MAX_HISTORY_LENGTH), dtype=bool

        # Step 3: Cast to float32 if needed
        #mask = tf.cast(mask, dtype='float32')  # Shape: (batch_size, MAX_HISTORY_LENGTH), dtype=float32

        # Fastformer
        #hidden_emb = self.fastformer([news_vectors, news_vectors, mask, mask])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        #hidden_emb = self.nrms_layer([news_vectors, news_vectors, mask, mask])


        #news_vectors = TimeDistributed(self.news_encoder_layer)(inputs)  # [batch, H, D]
        #mask_h = tf.reduce_any(tf.not_equal(inputs, 0), axis=-1)         # bool, shape=(batch, H)

        #hidden_emb = self.nrms_layer([news_vectors, news_vectors, mask_h, mask_h])

        news_vecs = TimeDistributed(self.news_encoder_layer)(inputs)  # [batch, H, D]

        # 2) bool mask over history positions
        mask_h = tf.reduce_any(tf.not_equal(inputs, 0), axis=-1)       # bool [batch, H]

        # 3) self-attend over history with that mask
        hidden_emb = self.nrms_layer([news_vecs, news_vecs, mask_h, mask_h])    # [batch, H, D]

        hidden_emb = self.dropout(hidden_emb)
        hidden_emb = self.layer_norm(hidden_emb)

        # Attention-based Pooling over history
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        user_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)

        return user_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(UserEncoderNRMS, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'news_encoder_layer': tf.keras.utils.serialize_keras_object(self.news_encoder_layer),
        })
        return config
    @classmethod
    def from_config(cls, config):
        # Extract the serialized news_encoder_layer config
        news_encoder_config = config.pop("news_encoder_layer")
        # Reconstruct the news_encoder_layer instance
        news_encoder_layer = tf.keras.utils.deserialize_keras_object(
            news_encoder_config, custom_objects={'NewsEncoderNRMS': NewsEncoderNRMS}
        )
        return cls(news_encoder_layer, **config)



@register_keras_serializable()
class TimedNRMSBlock(Layer):
    def __init__(self, nb_head, size_per_head, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(
            num_heads=nb_head,
            key_dim=size_per_head,
            dropout=dropout_rate,
            name=self.name + "_mha"
        )
        self.dropout = Dropout(dropout_rate, name=self.name + "_dropout")

    def call(self, inputs):
        # unpack
        timings = {}
        t0 = time.perf_counter()
        if len(inputs) == 4:
            Q_seq, K_seq, Q_mask_2d, K_mask_2d = inputs
        else:
            Q_seq, K_seq = inputs
            Q_mask_2d = K_mask_2d = None

        # if they gave you a 2D boolean mask, expand it to 3D:
        #   mask_2d: (batch, seq_len)
        #   ⇒ mask_3d: (batch, seq_len, seq_len)
        if K_mask_2d is not None:
            # make sure it's boolean
            K_mask_2d = tf.cast(K_mask_2d, tf.bool)
            # row‐ and column‐wise AND to get a [batch, q_len, k_len] mask
            mask_3d = tf.logical_and(
                tf.expand_dims(K_mask_2d, axis=1),    # (batch,1,k_len)
                tf.expand_dims(K_mask_2d, axis=2)     # (batch,q_len,1)
            )
        else:
            mask_3d = None
        timings['make_mask'] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        attn_out = self.mha(
            query=Q_seq,
            value=K_seq,
            key=K_seq,
            attention_mask=mask_3d,    # now shape=(batch, q_len, k_len)
            return_attention_scores=False
        )
        timings['mha'] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        output = self.dropout(attn_out)
        timings['drop'] = (time.perf_counter() - t0) * 1e3
        print_to_file({k: f"{v:.1f}ms" for k, v in timings.items()}, 'logs/timings.log')

        return output

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "nb_head": self.mha.num_heads,
            "size_per_head": self.mha.key_dim,
            "dropout_rate": self.dropout.rate,
        })
        return cfg

@register_keras_serializable()
class TimedFastformer(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        super(TimedFastformer, self).__init__(**kwargs)
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head

        self.WQ = None
        self.WK = None
        self.WV = None
        self.WO = None

    def build(self, input_shape):
        self.WQ = Dense(self.output_dim, use_bias=False, name='WQ')
        self.WK = Dense(self.output_dim, use_bias=False, name='WK')
        self.WV = Dense(self.output_dim, use_bias=False, name='WV')
        self.WO = Dense(self.output_dim, use_bias=False, name='WO')
        super(TimedFastformer, self).build(input_shape)

    def call(self, inputs):
        timings = {}
        t0 = time.perf_counter()
        if len(inputs) == 2:
            Q_seq, K_seq = inputs
            Q_mask = None
            K_mask = None
        elif len(inputs) == 4:
            Q_seq, K_seq, Q_mask, K_mask = inputs
        timings['make_mask'] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        batch_size = tf.shape(Q_seq)[0]
        seq_len = tf.shape(Q_seq)[1]

        # Linear projections
        Q = self.WQ(Q_seq)  # Shape: (batch_size, seq_len, output_dim)
        K = self.WK(K_seq)  # Shape: (batch_size, seq_len, output_dim)
        V = self.WV(K_seq)  # Shape: (batch_size, seq_len, output_dim)

        # Reshape for multi-head attention
        Q = tf.reshape(Q, (batch_size, seq_len, self.nb_head, self.size_per_head))
        K = tf.reshape(K, (batch_size, seq_len, self.nb_head, self.size_per_head))
        V = tf.reshape(V, (batch_size, seq_len, self.nb_head, self.size_per_head))

        # Compute global query and key
        global_q = tf.reduce_mean(Q, axis=1, keepdims=True)  # (batch_size, 1, nb_head, size_per_head)
        global_k = tf.reduce_mean(K, axis=1, keepdims=True)  # (batch_size, 1, nb_head, size_per_head)

        # Compute attention weights
        weights = global_q * K + global_k * Q  # (batch_size, seq_len, nb_head, size_per_head)
        weights = tf.reduce_sum(weights, axis=-1)  # (batch_size, seq_len, nb_head)
        weights = tf.nn.softmax(weights, axis=1)  # Softmax over seq_len

        # Apply attention weights to values
        weights = tf.expand_dims(weights, axis=-1)  # (batch_size, seq_len, nb_head, 1)
        context = weights * V  # (batch_size, seq_len, nb_head, size_per_head)

        # Combine heads
        context = tf.reshape(context, (batch_size, seq_len, self.output_dim))

        # Final projection
        output = self.WO(context)  # (batch_size, seq_len, output_dim)
        timings['fastformer_mha'] = (time.perf_counter() - t0) * 1e3
        print_to_file({k: f"{v:.1f}ms" for k, v in timings.items()}, 'logs/timings.log')

        return output  # Output shape: (batch_size, seq_len, output_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super(TimedFastformer, self).get_config()
        config.update({
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head
        })
        return config


@register_keras_serializable()
class TimedNewsEncoderNRMS(Layer):
    def __init__(self, vocab_size, embedding_dim=256, dropout_rate=0.2, nb_head=8, size_per_head=32, embedding_layer=None, **kwargs):
        super(TimedNewsEncoderNRMS, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.nb_head = nb_head
        self.size_per_head = size_per_head

        # Define sub-layers
        self.embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name='embedding_layer'
        )
        self.dropout = Dropout(self.dropout_rate)
        self.dense = Dense(1)
        self.softmax = Softmax(axis=1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

        #self.TimedFastformer_layer = TimedFastformer(nb_head=self.nb_head, size_per_head=self.size_per_head, name='TimedFastformer_layer')
        self.nrms_layer = TimedNRMSBlock(
            nb_head=self.nb_head,
            size_per_head=self.size_per_head,
            dropout_rate=self.dropout_rate,
            name="nrms_layer"
        )
    def build(self, input_shape):
        super(TimedNewsEncoderNRMS, self).build(input_shape)

    def call(self, inputs):
        # Create mask

        timings = {}
        t0 = time.perf_counter()
        mask = tf.not_equal(inputs, 0)
        timings['make_mask'] = (time.perf_counter() - t0) * 1e3
        # Embedding
        t0 = time.perf_counter()
        title_emb = self.embedding_layer(inputs)  # Shape: (batch_size, seq_len, embedding_dim)
        title_emb = self.dropout(title_emb)
        timings['title_emb'] = (time.perf_counter() - t0) * 1e3

        # TimedFastformer
        #hidden_emb = self.TimedFastformer_layer([title_emb, title_emb, mask, mask])  # Shape: (batch_size, seq_len, embedding_dim)
        #mha_mask = tf.reshape(mask, [tf.shape(mask)[0], 1, 1, tf.shape(mask)[1]])

        # 4) apply your NRMS self‐attention block
        t0 = time.perf_counter()
        hidden_emb = self.nrms_layer([title_emb, title_emb, mask, mask])
        timings['news_encoder_nrms'] = (time.perf_counter() - t0) * 1e3
    



        t0 = time.perf_counter()
        hidden_emb = self.dropout(hidden_emb)
        timings['drop'] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        # Attention-based Pooling
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, seq_len, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, seq_len, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, seq_len, embedding_dim)
        news_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)
        timings['attn_pool'] = (time.perf_counter() - t0) * 1e3
        print_to_file({k: f"{v:.1f}ms" for k, v in timings.items()}, 'logs/timings.log')

        return news_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(TimedNewsEncoderNRMS, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head,
            'embedding_layer': tf.keras.utils.serialize_keras_object(self.embedding_layer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_layer_config = config.pop('embedding_layer', None)
        embedding_layer = tf.keras.layers.deserialize(embedding_layer_config) if embedding_layer_config else None
        return cls(embedding_layer=embedding_layer, **config)


@register_keras_serializable()
class TimedUserEncoderNRMS(Layer):
    def __init__(self, news_encoder_layer, embedding_dim=256, **kwargs):
        super(TimedUserEncoderNRMS, self).__init__(**kwargs)
        self.news_encoder_layer = news_encoder_layer
        self.embedding_dim = embedding_dim
        self.dropout = Dropout(0.2)
        self.layer_norm = LayerNormalization()
        #self.TimedFastformer = TimedFastformer(nb_head=8, size_per_head=32, name='user_TimedFastformer')
        self.nrms_layer = TimedNRMSBlock(
            nb_head=8,
            size_per_head=32,
            name="user_nrms_layer"
        )
        self.dense = Dense(1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.softmax = Softmax(axis=1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

    def call(self, inputs):
        # inputs: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH)

        timings = {}
        t0 = time.perf_counter()
        news_vecs = TimeDistributed(self.news_encoder_layer)(inputs)  # [batch, H, D]
        timings['encode_hist'] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        # 2) bool mask over history positions
        mask_h = tf.reduce_any(tf.not_equal(inputs, 0), axis=-1)       # bool [batch, H]
        timings['make_mask'] = (time.perf_counter() - t0) * 1e3

        # 3) self-attend over history with that mask
        t0 = time.perf_counter()
        hidden_emb = self.nrms_layer([news_vecs, news_vecs, mask_h, mask_h])    # [batch, H, D]
        timings['user_encoder_nrms'] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        hidden_emb = self.dropout(hidden_emb)
        hidden_emb = self.layer_norm(hidden_emb)
        timings['drop_norm'] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        # Attention-based Pooling over history
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        user_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)
        timings['attn_pool'] = (time.perf_counter() - t0) * 1e3
        print_to_file({k: f"{v:.1f}ms" for k, v in timings.items()}, 'logs/timings.log')

        return user_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(TimedUserEncoderNRMS, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'news_encoder_layer': tf.keras.utils.serialize_keras_object(self.news_encoder_layer),
        })
        return config
    @classmethod
    def from_config(cls, config):
        # Extract the serialized news_encoder_layer config
        news_encoder_config = config.pop("news_encoder_layer")
        # Reconstruct the news_encoder_layer instance
        news_encoder_layer = tf.keras.utils.deserialize_keras_object(
            news_encoder_config, custom_objects={'TimedNewsEncoderNRMS': TimedNewsEncoderNRMS}
        )
        return cls(news_encoder_layer, **config)



@register_keras_serializable()
class TimedNewsEncoder(Layer):
    def __init__(self, vocab_size, embedding_dim=256, dropout_rate=0.2, nb_head=8, size_per_head=32, embedding_layer=None, **kwargs):
        super(TimedNewsEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.nb_head = nb_head
        self.size_per_head = size_per_head

        # Define sub-layers
        self.embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name='embedding_layer'
        )
        self.dropout = Dropout(self.dropout_rate)
        self.dense = Dense(1)
        self.softmax = Softmax(axis=1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

        self.fastformer_layer = TimedFastformer(nb_head=self.nb_head, size_per_head=self.size_per_head, name='fastformer_layer')

    def build(self, input_shape):
        super(TimedNewsEncoder, self).build(input_shape)

    def call(self, inputs):
        # Create mask
        timings = {}
        t0 = time.perf_counter()
        mask = tf.cast(tf.not_equal(inputs, 0), dtype='float32')  # Shape: (batch_size, seq_len)
        timings['make_mask'] = (time.perf_counter() - t0) * 1e3

        # Embedding
        t0 = time.perf_counter()
        title_emb = self.embedding_layer(inputs)  # Shape: (batch_size, seq_len, embedding_dim)
        title_emb = self.dropout(title_emb)
        timings['title_emb'] = (time.perf_counter() - t0) * 1e3

        # Fastformer
        t0 = time.perf_counter()
        hidden_emb = self.fastformer_layer([title_emb, title_emb, mask, mask])  # Shape: (batch_size, seq_len, embedding_dim)
        timings['news_encoder_fastformer'] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        hidden_emb = self.dropout(hidden_emb)
        timings['drop'] = (time.perf_counter() - t0) * 1e3

        # Attention-based Pooling
        t0 = time.perf_counter()
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, seq_len, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, seq_len, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, seq_len, embedding_dim)
        news_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)
        timings['attn_pool'] = (time.perf_counter() - t0) * 1e3
        print_to_file({k: f"{v:.1f}ms" for k, v in timings.items()}, 'logs/timings.log')

        return news_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(TimedNewsEncoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head,
            'embedding_layer': tf.keras.utils.serialize_keras_object(self.embedding_layer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_layer_config = config.pop('embedding_layer', None)
        embedding_layer = tf.keras.layers.deserialize(embedding_layer_config) if embedding_layer_config else None
        return cls(embedding_layer=embedding_layer, **config)



@register_keras_serializable()
class TimedUserEncoder(Layer):
    def __init__(self, news_encoder_layer, embedding_dim=256, **kwargs):
        super(TimedUserEncoder, self).__init__(**kwargs)
        self.news_encoder_layer = news_encoder_layer
        self.embedding_dim = embedding_dim
        self.dropout = Dropout(0.2)
        self.layer_norm = LayerNormalization()
        self.fastformer = TimedFastformer(nb_head=8, size_per_head=32, name='user_fastformer')
        self.dense = Dense(1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.softmax = Softmax(axis=1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)
    #@tf.function
    def call(self, inputs):
        # inputs: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH)
        timings = {}
        # Encode each news article in the history
        t0 = time.perf_counter()
        news_vectors = TimeDistributed(self.news_encoder_layer)(inputs)  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        timings['encode_hist'] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        mask = tf.not_equal(inputs, 0)  # Shape: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH), dtype=bool
        mask = tf.reduce_any(mask, axis=-1)  # Shape: (batch_size, MAX_HISTORY_LENGTH), dtype=bool
        mask = tf.cast(mask, dtype='float32')  # Shape: (batch_size, MAX_HISTORY_LENGTH), dtype=float32
        timings['make_mask'] = (time.perf_counter() - t0) * 1e3

        # Fastformer
        t0 = time.perf_counter()
        hidden_emb = self.fastformer([news_vectors, news_vectors, mask, mask])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        timings['user_encoder_fastformer'] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        hidden_emb = self.dropout(hidden_emb)
        hidden_emb = self.layer_norm(hidden_emb)
        timings['drop_norm'] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        # Attention-based Pooling over history
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        user_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)
        timings['attn_pool'] = (time.perf_counter() - t0) * 1e3
        print_to_file({k: f"{v:.1f}ms" for k, v in timings.items()}, 'logs/timings.log')
        #tf.print({k: f"{v:.1f}ms" for k, v in timings.items()})

        return user_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(TimedUserEncoder, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'news_encoder_layer': tf.keras.utils.serialize_keras_object(self.news_encoder_layer),
        })
        return config
    @classmethod
    def from_config(cls, config):
        # Extract the serialized news_encoder_layer config
        news_encoder_config = config.pop("news_encoder_layer")
        # Reconstruct the news_encoder_layer instance
        news_encoder_layer = tf.keras.utils.deserialize_keras_object(
            news_encoder_config, custom_objects={'TimedNewsEncoder': TimedNewsEncoder}
        )
        return cls(news_encoder_layer, **config)


def build_nrms_model(vocab_size, max_title_length=30, max_history_length=50, embedding_dim=256, nb_head=8, size_per_head=32, dropout_rate=0.2, timed=False):
    # Define Inputs
    history_input = Input(shape=(max_history_length, max_title_length), dtype='int32', name='history_input')
    candidate_input = Input(shape=(max_title_length,), dtype='int32', name='candidate_input')

    # Instantiate NewsEncoder Layer
    if timed:
        news_encoder_layer = TimedNewsEncoderNRMS(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            nb_head=nb_head,
            size_per_head=size_per_head,
            name='news_encoder'
        )
    else:
        news_encoder_layer = NewsEncoderNRMS(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            nb_head=nb_head,
            size_per_head=size_per_head,
            name='news_encoder'
        )

    # Encode Candidate News
    candidate_vector = news_encoder_layer(candidate_input)  # Shape: (batch_size, embedding_dim)

    # Encode User History
    if timed:
        user_vector = TimedUserEncoderNRMS(news_encoder_layer, embedding_dim=embedding_dim, name='user_encoder')(history_input)  # Shape: (batch_size, embedding_dim)
    else:
        user_vector = UserEncoderNRMS(news_encoder_layer, embedding_dim=embedding_dim, name='user_encoder')(history_input)  # Shape: (batch_size, embedding_dim)

    # Scoring Function: Dot Product between User and Candidate Vectors
    score = Dot(axes=-1)([user_vector, candidate_vector])  # Shape: (batch_size, 1)
    score = Activation('sigmoid')(score)  # Shape: (batch_size, 1)

    # Build Model
    model = Model(inputs={'history_input': history_input, 'candidate_input': candidate_input}, outputs=score)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='AUC')
        ]
    )

    return model


def build_and_load_weights(weights_file):
     print("""Building model: build_model(
         vocab_size={vocab_size},
         max_title_length={max_title_length},
         max_history_length={max_history_length},
         embedding_dim=256,
         nb_head=8,
         size_per_head=32,
         dropout_rate=0.2
     )""")
     model = build_model(
         vocab_size=vocab_size,
         max_title_length=max_title_length,
         max_history_length=max_history_length,
         embedding_dim=256,
         nb_head=8,
         size_per_head=32,
         dropout_rate=0.2
     )
     input_shapes = {
         'history_input': (None, max_history_length, max_title_length),
         'candidate_input': (None, max_title_length)
     }
 
     dummy_history_input = np.zeros((1, 50, 30), dtype=np.int32)
     dummy_candidate_input = np.zeros((1, 30), dtype=np.int32)
 
     # Build the model by passing dummy data
     model.predict({'history_input': dummy_history_input, 'candidate_input': dummy_candidate_input})
     #model.build(input_shapes)
     model.load_weights(weights_file)
     return model
    
def glob_size_variants(template, root = Path(".")):
    glob_pattern = re.sub(r'\b(?:small|large)\b', '*', template)
    return sorted(root.glob(glob_pattern))
def train_cluster_models(clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters, batch_size=64, epochs=5, load_models=[], retrain=False, size='large', train_part=0.8,
    model_base='', dataset_size='small', dataset='train', test_size=-0.1, load_all_sizes=False):
    models = {}
    model_dir = "models"    
    for cluster in range(num_clusters):
        if model_base == '':
            m_name = f'fastformer_cluster_{cluster}_{size}_full_balanced_1_epoch'
        else:
            m_name = f'fastformer_cluster_{cluster}_{model_base}'
        weights_file = f'{model_dir}/{m_name}.weights.h5'
        model_file = f'{model_dir}/{m_name}.keras'
        model_h5_file = f'{model_dir}/{m_name}.h5'
        finedtuned_model_h5_file = f'{model_dir}/{m_name}_tuned_{dataset}_{dataset_size}_tsize_{test_size}.h5'
        model_hdf5_file = f'{model_dir}/{m_name}.hdf5'
        model_json_file = f'{model_dir}/{m_name}.json'
        #if cluster in load_models: # load_models should be list of number indicating which models to load and not train
        #    print(f"\nLoading model for Cluster {cluster} from {model_file}")
        #    local_model_path = hf_hub_download(
        #        repo_id=f"Teemu5/news",
        #        filename=model_file,
        #        local_dir=model_dir
        #    )
        if dataset == 'valid' and os.path.exists(finedtuned_model_h5_file):
            model_path = finedtuned_model_h5_file #
        else:
            model_path = model_h5_file
        if os.path.exists(model_path) and not retrain:
            print(f"Loading model: {model_path}")
            print(tf.__version__)
            print(keras.__version__)
            if load_all_sizes:
                for p in glob_size_variants(model_path):
                    if f"{cluster}_large" in p and "tuned" not in p:
                        continue # large models should be finetuned so skip non tuned large models
                    log_print(f"loading model p:{p}")
                    with custom_object_scope({'UserEncoder': UserEncoder, 'NewsEncoder': NewsEncoder}):
                        model = tf.keras.models.load_model(p)
                        key = f"{cluster}_large" if f"{cluster}_large" in p else f"{cluster}_small"
                        models[f"{cluster}_{size}"] = model
                continue # load all models not training so continue
            else:
                with custom_object_scope({'UserEncoder': UserEncoder, 'NewsEncoder': NewsEncoder}):
                    model = tf.keras.models.load_model(model_path)#build_and_load_weights(weights_file)
                    models[f"{cluster}_{size}"] = model
            #model.save(model_file)
            #print(f"Saved model for Cluster {cluster} into {model_file}.")
            if model_path == finedtuned_model_h5_file or size == 'small':
                continue
            elif dataset == 'valid': # Conditionally finetuning existing model
                model_h5_file = finedtuned_model_h5_file
        
        print(f"\nTraining model for Cluster {cluster} into {model_h5_file}")
        # Retrieve training and validation data
        train_data = clustered_data[cluster]['train']
        val_data = clustered_data[cluster]['val']
        if model_h5_file == finedtuned_model_h5_file: # fine tuning on non clusterized data
            train_parts = [clustered_data[c]['train'] for c in clustered_data]
            val_parts   = [clustered_data[c]['val']   for c in clustered_data]
            train_data = pd.concat(train_parts, ignore_index=True)
            val_data   = pd.concat(val_parts,   ignore_index=True)

        print(f"Cluster {cluster} - Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

        # Create data generators
        train_generator = DataGenerator(train_data, batch_size, max_history_length, max_title_length)
        val_generator = DataGenerator(val_data, batch_size, max_history_length, max_title_length)

        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)

        # Build model
        model = build_model(
            vocab_size=vocab_size,
            max_title_length=max_title_length,
            max_history_length=max_history_length,
            embedding_dim=256,
            nb_head=8,
            size_per_head=32,
            dropout_rate=0.2
        )
        print(model.summary())

        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_AUC',
            patience=2,
            mode='max',
            restore_best_weights=True
        )
        csv_logger = CSVLogger(f'training_log_cluster_{cluster}.csv', append=True)
        model_checkpoint = ModelCheckpoint(
            f'best_model_cluster_{cluster}.keras',
            monitor='val_AUC',
            mode='max',
            save_best_only=True
        )
        print("\nlabel balance (0 vs 1):")
        print(train_df['Label'].value_counts())
        class_weight = get_class_weights(train_df['Label'])
        print("Class weights:", class_weight)
        logging.info(f"Class weights: {class_weight}")
        # Train the model
        model.fit(
            train_generator,
            epochs=epochs,
            #steps_per_epoch=steps_per_epoch,
            #validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[early_stopping, csv_logger, model_checkpoint],
            class_weight=class_weight
        )

        # Save model weights
        #model.save_weights(weights_file)
        #print(f"Saved model weights for Cluster {cluster} into {weights_file}.")
        model.save(model_h5_file)
        print(f"Saved h5 model for Cluster {cluster} into {model_h5_file}.")
        #model.save(model_file)
        #print(f"Saved model for Cluster {cluster} into {model_file}.")

        # Store the model
        models[f"{cluster}_{size}"] = model
        # Clear memory
        del train_data, val_data, train_generator, val_generator, model
        import gc
        gc.collect()
    print("Returning models list")
    print(models)
    return models

import math, os, tensorflow as tf
from tensorflow.keras.callbacks import Callback

class SaveEveryN(tf.keras.callbacks.Callback):
    def __init__(self, n_batches, tmpl):
        super().__init__()
        self.n_batches, self.tmpl = n_batches, tmpl
        self.b_seen = 0

    def on_train_batch_end(self, batch, logs=None):
        self.b_seen += 1
        if self.b_seen % self.n_batches == 0:
            fname = self.tmpl.format(batch=self.b_seen)
            self.model.save(fname, save_format="h5")       
            print(f"[ckpt] saved {fname}")


def resume_or_build_h5(model_dir, build_fn, steps_per_epoch, timed=False):
    pattern = os.path.join(model_dir, "*_ckpt_batch*.h5")
    ckpts   = glob.glob(pattern)
    if not ckpts:
        return build_fn(), 0

    def _batch_id(p): 
        m = re.search(r"_batch(\d+)\.h5$", p)
        return int(m.group(1)) if m else -1
    latest = max(ckpts, key=_batch_id)
    batches_done = _batch_id(latest)
    start_epoch  = batches_done // steps_per_epoch

    print(f"Resuming from {latest} (batch {batches_done}, epoch {start_epoch})")
    model = load(latest, custom_objects=None, timed=timed)
    return model, start_epoch


def train_global_model(
    train_data,
    tokenizer,
    vocab_size,
    max_history_length,
    max_title_length,
    dataset_size,
    batch_size=128,
    epochs=5,
    val_data=None,
    retrain=False,
    load_best_model=False,
    load_checkpoint_model=False, model_base='', model_arc_type="fastformer", timed=False
):

    if model_base == '':
        model_file_prefix=f"{model_arc_type}_global_{dataset_size}_balanced_{epochs}_epochs"
    else:
        model_file_prefix = f'{model_arc_type}_global_{model_base}'
    #model_file_prefix=f"{model_arc_type}_global_{dataset_size}_balanced_{epochs}_epochs"
    model_dir="models"
    weights_file = f'{model_dir}/{model_file_prefix}.weights.h5'
    best_model = f'{model_dir}/best_model_{model_file_prefix}.h5'
    h5_file = f'{model_dir}/{model_file_prefix}.h5'
    keras_file   = f'{model_dir}/{model_file_prefix}.keras'

    # if the .keras file already exists, load and return it
    if not retrain and os.path.exists(h5_file):
        if load_best_model:
            load_file = best_model
        else:
            load_file = h5_file
        custom_objects = {
            'UserEncoder': UserEncoder,
            'NewsEncoder': NewsEncoder,
        }
        if timed:
            custom_objects = {
                'UserEncoder': TimedUserEncoder,
                'NewsEncoder': TimedNewsEncoder,
            }
        if model_arc_type == "nrms":
            custom_objects = {
                'UserEncoderNRMS': UserEncoderNRMS,
                'NewsEncoderNRMS': NewsEncoderNRMS,
                'UserEncoder':    UserEncoder,
                'NewsEncoder':    NewsEncoder,
                'Fastformer':     Fastformer,
                'SqueezeLayer':   SqueezeLayer,
                'ExpandDimsLayer':ExpandDimsLayer,
                'SumPooling':     SumPooling,
                'MaskLayer':      MaskLayer,
                'NotEqual':       tf.not_equal,
            }
            if timed:
                custom_objects = {
                    'UserEncoderNRMS': TimedUserEncoderNRMS,
                    'NewsEncoderNRMS': TimedNewsEncoderNRMS,
                    'UserEncoder':    TimedUserEncoder,
                    'NewsEncoder':    TimedNewsEncoder,
                    'Fastformer':     TimedFastformer,
                    'SqueezeLayer':   SqueezeLayer,
                    'ExpandDimsLayer':ExpandDimsLayer,
                    'SumPooling':     SumPooling,
                    'MaskLayer':      MaskLayer,
                    'NotEqual':       tf.not_equal,
                }
        model = load(load_file, custom_objects, timed=timed)
        print(f"Loaded existing global model from {keras_file}")
        return model
    #train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

    train_generator = DataGenerator(train_data, batch_size=batch_size, max_history_length=max_history_length, max_title_length=max_title_length)
    #if val_data:
    #    val_generator = DataGenerator(val_data, batch_size=batch_size, max_history_length=max_history_length, max_title_length=max_title_length)
    steps_per_epoch = len(train_generator)
    if model_arc_type == "fastformer":
        model, start_epoch = resume_or_build_h5(
            model_dir=model_dir,
            build_fn=lambda: build_model(
                vocab_size=vocab_size,
                max_title_length=max_title_length,
                max_history_length=max_history_length,
                embedding_dim=256,
                nb_head=8,
                size_per_head=32,
                dropout_rate=0.2,
                timed=timed
            ),
            steps_per_epoch=steps_per_epoch
        )
    elif model_arc_type == "nrms":
        start_epoch = 0
        dg = DataGeneratorNRMS(train_data, batch_size, max_history_length=50, max_title_length=30)

        model = build_nrms_model(vocab_size,
            max_title_length=30,
            max_history_length=50,
            embedding_dim=256,
            nb_head=8,
            size_per_head=32,
            dropout_rate=0.2,
            timed=timed
        )

    print(model.summary())
    if load_checkpoint_model:
        return model

    early_stopping   = EarlyStopping(
        monitor='val_AUC',
        patience=2,
        mode='max',
        restore_best_weights=True
    )
    csv_logger = CSVLogger(f'{model_file_prefix}_training_log.csv', append=True)
    model_checkpoint = ModelCheckpoint(
        best_model,
        monitor='val_AUC',
        mode='max',
        save_best_only=True
    )
    print("label balance (0 vs 1):")
    print(train_df['Label'].value_counts())
    class_weight = get_class_weights(train_df['Label'])
    print("Class weights:", class_weight)
    logging.info(f"Class weights: {class_weight}")

    save_every = max(1, math.floor(steps_per_epoch * 0.1))
    batch_ckpt_tmpl = (
        f"{model_dir}/{model_file_prefix}_ckpt_batch{{batch:06d}}.keras"
    )
    save_every_n = SaveEveryN(save_every, batch_ckpt_tmpl)
    if model_arc_type == "nrms":
        model.fit(dg,
            epochs=epochs,
            initial_epoch=start_epoch,
            steps_per_epoch=steps_per_epoch,
            #validation_data=val_generator,
            callbacks=[early_stopping, csv_logger, model_checkpoint, save_every_n],
            class_weight=class_weight
        )
    else:
        model.fit(
            train_generator,
            epochs=epochs,
            initial_epoch=start_epoch,
            steps_per_epoch=steps_per_epoch,
            #validation_data=val_generator,
            callbacks=[early_stopping, csv_logger, model_checkpoint, save_every_n],
            class_weight=class_weight
        )

    model.save_weights(weights_file)
    print(f"Saved weights to {weights_file}")
    model.save(h5_file)
    print(f"Saved H5 model to {h5_file}")
    model.save(keras_file)
    print(f"Saved full model to {keras_file}")

    return model

import pandas as pd, numpy as np, os, time, json, pathlib

def _flush(buffer, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(buffer).to_parquet(path, engine="fastparquet",
                                    compression="snappy",
                                    index=False,
                                    append=os.path.exists(path))


import time
import tracemalloc
import psutil
import torch
import pandas as pd

def regenerate_metrics_from_parquet(parquet_path: str,
                                    decision_threshold : float = 0.5,
                                    tune_threshold     : bool  = False,
                                    k_values           : tuple = (5,10,20,50,100),
                                    slate_key          : str   = "ImpressionID",
                                    store_metrics_dir  : str   = "base_preds",
                                    store_metrics_file : str   = "metrics.json"):
    """
    Re-compute the JSON metrics file for an *existing* prediction parquet.
      • `parquet_path` must contain y_true and y_pred columns (and row_id).
      • `tune_threshold=True` will search the threshold that maximises F1,
        exactly as `evaluate_with_generator` does.
    The regenerated metrics are saved to
        {store_metrics_dir}_{threshold:.4f}_<store_metrics_file>
    and also returned as a dict.
    """
    t0 = time.perf_counter()
    df = pd.read_parquet(parquet_path)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("Parquet must contain 'y_true' and 'y_pred' columns")

    y_true = df["y_true"].to_numpy(dtype="int8")
    y_pred = df["y_pred"].to_numpy(dtype="float32")

    if tune_threshold:
        p, r, thr = precision_recall_curve(y_true, y_pred)
        f1 = 2 * p * r / (p + r + 1e-12)
        decision_threshold = thr[np.nanargmax(f1)]
        print(f"F1-optimal threshold = {decision_threshold:0.4f}")

    bin_pred = (y_pred >= decision_threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, bin_pred, average="binary", zero_division=0)

    # basic classification metrics
    metrics = dict(
        samples   = int(len(y_true)),
        AUC       = float(roc_auc_score(y_true, y_pred)),
        AP        = float(average_precision_score(y_true, y_pred)),
        precision = float(p), recall=float(r), F1=float(f1),
        threshold = float(decision_threshold),
        runtime_s = time.perf_counter() - t0
    )

    # add ranking metrics (uses your helper)
    rank = evaluate_parquet_scores(parquet_path, k_values=[5,10,20,50,100])
    metrics.update(rank)

    # write metrics file
    thr_tag = f"{decision_threshold:0.4f}"
    out_path = Path(f"{store_metrics_dir}/{thr_tag}_{store_metrics_file}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=4)
    print(f"metrics written to {out_path}")

    return metrics

def evaluate_with_generator(model,
                            eval_df,
                            batch_size      = 256,
                            max_history_len = 50,
                            max_title_len   = 30,
                            store_path      = None,
                            flush_every     = 0.10,
                            verbose         = 1,
                            store_metrics_dir="base_preds",
                            store_metrics_file="metrics.json",
                            slate_key = "ImpressionID",
                            threshold = 0.5,
                            tune_threshold = False,
                            model_key = "model",
                            timed=False):
    decision_threshold = threshold
    log_print(f"evaluating model {model} on DataFrame with {len(eval_df)} rows and columns {list(eval_df.columns)}")
    log_print(f"store_path:{store_path} and store_metrics_dir: {store_metrics_dir}")
    start_time = time.perf_counter() # mark start time
    tracemalloc.start() # begin Python memory tracing
    process = psutil.Process()
    proc = process
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats() # zero out GPU memory peaks
    # Containers for batch‐level stats
    batch_latencies = []


    gen = DataGenerator(eval_df, batch_size,
        max_history_length=max_history_len,
        max_title_length=max_title_len)

    total_batches   = len(gen)
    flush_interval  = max(1, int(total_batches * flush_every))
    buf             = []
    written, seen   = 0, 0
    y_true_all, y_pred_all = [], []

    iterator = tqdm(range(total_batches), desc="Eval", disable=(verbose==0))

    for b in iterator:
        t0 = time.perf_counter()

        start = b * batch_size
        end   = min((b + 1) * batch_size, len(eval_df))
        batch_df  = eval_df.iloc[start:end]

        X, y_true = gen[b]
        if X is None:
            continue



        #logdir = f"./logs/eval/{model_key}"
        #out_dir = Path(logdir)
        #out_dir.mkdir(exist_ok=True)

        from tensorflow.keras.callbacks import TensorBoard
        from tensorflow.python.profiler import profiler_v2 as profiler

        logdir = "./logs/profile_cpu_only"
        """
        tf.profiler.experimental.start(
            logdir,
            options=tf.profiler.experimental.ProfilerOptions(
                # profile Python & TF ops on the host:
                host_tracer_level=2,
                python_tracer_level=1,
                # disable GPU/CUPTI tracing:
                device_tracer_level=0
            )
        )
        """
        # create the TensorBoard callback with profiling enabled for batches 2–4
        #tb_cb = tf.keras.callbacks.TensorBoard(
        #    log_dir=logdir,
        #    profile_batch=(2, 4)
        #)
        #tf.profiler.experimental.start('logs/trace')
        """
        import tensorflow as tf
        tf.config.run_functions_eagerly(True)

        for layer in model.layers:
            orig_call = layer.call
            def make_timed(orig_call):
                @functools.wraps(orig_call)
                def timed(self, inputs, *args, **kwargs):
                    t0 = time.perf_counter()
                    out = orig_call(inputs, *args, **kwargs)
                    dt = (time.perf_counter() - t0) * 1000
                    print(f"{self.name:20s} → {dt:6.1f} ms")
                    return out
                return timed

            layer.call = make_timed(orig_call).__get__(layer, layer.__class__)
        """
        #inp, _ = next(iter(X))   # get one batch of inputs,labels
        #_ = model.predict_on_batch(inp)
        #return s
        y_pred = model.predict(X,
                               batch_size=batch_size,
                               verbose=1,
                               #callbacks=[tb_cb]
                            ).squeeze()
        #tf.profiler.experimental.stop()
        #tf.profiler.experimental.stop()
        t1 = time.perf_counter()
        batch_latencies.append(t1 - t0)


        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        if store_path is not None:
            #for lbl, pred, impr in zip(y_true, y_pred, batch_df["ImpressionID"]):
            #    buf.append({'row_id': seen, "ImpressionID": int(impr),
            for lbl, pred, impr in zip(y_true, y_pred, eval_df[slate_key].iloc[start:end]):
                buf.append({'row_id': seen,
                            slate_key: impr,
                            'y_true': float(lbl),
                            'y_pred': float(pred)})
                seen += 1
            if len(buf) >= flush_interval * batch_size:
                _flush(buf, store_path)
                written += len(buf); buf.clear()

        if (b + 1) % flush_interval == 0 and verbose:
            y_true_so_far = np.concatenate(y_true_all, dtype="float32")
            y_pred_so_far = np.concatenate(y_pred_all, dtype="float32")
            auc_so_far = roc_auc_score(y_true_so_far, y_pred_so_far)
            p, r, f, _ = precision_recall_fscore_support(
                            y_true_so_far,
                            (y_pred_so_far >= 0.5).astype(int),
                            average="binary",
                            zero_division=0)
            print(f"[{b+1}/{total_batches}] AUC={auc_so_far:.4f} P={p:.4f} R={r:.4f} F1={f:.4f}")

    # flush leftovers
    if buf and store_path is not None:
        _flush(buf, store_path)
        written += len(buf)
    if verbose and store_path is not None:
        print(f"Stored {written:,} rows to {store_path}")


    total_time = time.perf_counter() - start_time

    peak_gpu_mem = None
    if torch.cuda.is_available():
        peak_gpu_mem = torch.cuda.max_memory_allocated()  # peak during run

    current_mem, peak_mem = tracemalloc.get_traced_memory()  # current & peak
    tracemalloc.stop()

    rss = process.memory_info().rss  # Resident Set Size


    y_true_all = np.concatenate(y_true_all, dtype="float32")
    y_pred_all = np.concatenate(y_pred_all, dtype="float32")

    if tune_threshold:
        p, r, thr = precision_recall_curve(y_true_all, y_pred_all)
        f1 = 2 * p * r / (p + r + 1e-12)
        decision_threshold = thr[np.nanargmax(f1)]
        log_print(f"F1-optimal threshold = {decision_threshold:0.4f}")

    binary_pred = (y_pred_all >= decision_threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_all, binary_pred, average="binary", zero_division=0)

    auc = roc_auc_score(y_true_all, y_pred_all)
    elapsed = total_time
    cur_mem = current_mem
    tracemalloc.stop()

    metrics = dict(samples = int(len(y_true_all)),
                AUC       = float(auc),
                precision = float(p), recall = float(r), F1 = float(f1),
                threshold = float(decision_threshold),
                runtime_s = elapsed,
                python_mem_current_bytes = cur_mem,
                python_mem_peak_bytes    = peak_mem,
                process_rss_bytes        = proc.memory_info().rss,
                gpu_peak_bytes           = torch.cuda.max_memory_allocated()
                                            if torch.cuda.is_available() else 0)

    if store_path is not None:
        metrics_k = evaluate_parquet_scores(store_path,
                                            k_values=[5,10,20,50,100])
        metrics.update(metrics_k)
    if timed:
        with open('logs/timings.log') as f:
            log_lines = [l.strip() for l in f if l.strip()]

        # parse lines like "{'make_mask': '0.8ms', 'mha': '16.4ms', ...}"
        times = defaultdict(list)
        pattern = re.compile(r"'(\w+)':\s*'([\d\.]+)ms'")
        for line in log_lines:
            for key, val in pattern.findall(line):
                times[key].append(float(val))

        # compute per-component averages
        avg_times = {comp: sum(vs) / len(vs) for comp, vs in times.items()}

        # 3) merge into your main metrics dict
        #    e.g. metrics['avg_make_mask_ms'] = 0.8, metrics['avg_mha_ms'] = 16.4, ...
        for comp, avg in avg_times.items():
            metrics[f'avg_{comp}_ms'] = avg

        # 4) (optional) also print a nice DataFrame
        df = (
            pd.DataFrame([
                {'component': comp, 'avg_time_ms': avg}
                for comp, avg in avg_times.items()
            ])
            .sort_values('avg_time_ms', ascending=False)
            .reset_index(drop=True)
        )
        print("\nPer-layer average timings:")
        print(df.to_string(index=False))
    thr_metrics_file = f"{decision_threshold:0.4f}_{store_metrics_file}"
    store_metrics_path = Path(f"{store_metrics_dir}/{thr_metrics_file}")
    print(f"write metrics to: {store_metrics_path}")
    # save metrics file
    with open(store_metrics_path, "w") as fp:
        json.dump(metrics, fp, indent=4)

    if verbose:
        print("Final metrics:", json.dumps(metrics, indent=2))

    return metrics

def load(load_this, custom_objects=None, max_history_length=50, max_title_length=30, timed=False):
    if timed:
        tf.config.run_functions_eagerly(True)
        tf.config.experimental.set_synchronous_execution(True)
    logging.info(f"Loading model: {load_this}")
    logging.info(f"custom_objects: {custom_objects}")
    print(f"Loading model: {load_this}")
    if custom_objects == None:
        custom_objects = {
            'UserEncoder': UserEncoder,
            'NewsEncoder': NewsEncoder,
        }
    import keras
    print(tf.__version__)
    print(tf.keras.__version__)
    print(keras.__version__)
    if custom_objects == None:
        model = tf.keras.models.load_model(load_this)
    else:
        #Load model within the custom object scope.
        from tensorflow.keras.utils import custom_object_scope
        with custom_object_scope(custom_objects):
            # Load without compiling to avoid optimizer issues
            model = tf.keras.models.load_model(load_this, compile=False)
            dummy_history = tf.zeros((1, max_history_length, max_title_length), dtype=tf.int32)
            dummy_candidate = tf.zeros((1, max_title_length), dtype=tf.int32)
            # Call the model once with a dictionary mapping inputs.
            _ = model({'history_input': dummy_history, 'candidate_input': dummy_candidate})

    print(f"{load_this} Model loaded successfully.")
    logging.info(f"Model Loaded {load_this}")
    return model

def train_category_models(category_train_dfs, vocab_size, max_history_length, max_title_length, batch_size=64, epochs=5, dataset_size='',
    train_only_new=True, train_fraction=1.0, load_best_model=True, model_base=''):
    #Train a model for each category in the category_train_dfs dict.
    model_dir = "models"
    category_models = {}
    for category, df in category_train_dfs.items():
        if model_base == '':
            model_file_prefix=f"fastformer_{dataset_size}_category_{category}_{epochs}epochs"
        else:
            model_file_prefix = f'fastformer_category_{category}_{model_base}'
        print(f"Training {category.upper()}")
        print(df)
        train_df, val_df = df["train"], df["val"]
        keras_model_save_path = f'{model_dir}/{model_file_prefix}.keras'
        model_save_path = f'{model_dir}/{model_file_prefix}.h5'
        best_model = f'{model_dir}/best_model_{model_file_prefix}.h5'
        keras_best_model = f'{model_dir}/{model_file_prefix}.keras'
        if train_only_new and os.path.exists(model_save_path):
            if load_best_model:
                category_models[category] = load(best_model)
            else:
                category_models[category] = load(model_save_path)
            continue
        print(f"Training model for category: {category}")
        #gpus = tf.config.list_physical_devices('GPU')
        #if gpus:
        #    for gpu in gpus:
        #        print(f"enable memory growth on gpu: {gpu}")
        #        tf.config.experimental.set_memory_growth(gpu, True)
        # Split into train/validation sets
        #train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
        train_data = train_df
        val_data = val_df
        print(f"Category '{category}': {len(train_data)} training samples, {len(val_data)} validation samples.")

        train_generator = DataGenerator(train_data, batch_size=batch_size, max_history_length=max_history_length, max_title_length=max_title_length)
        val_generator = DataGenerator(val_data, batch_size=batch_size, max_history_length=max_history_length, max_title_length=max_title_length)

        model = build_model(vocab_size, max_title_length, max_history_length, embedding_dim=256, nb_head=8, size_per_head=32, dropout_rate=0.2)
        model.summary(print_fn=lambda x: print(f"[{category}] {x}"))

        early_stopping = EarlyStopping(monitor='val_AUC', patience=2, mode='max', restore_best_weights=True)
        csv_logger = CSVLogger(f'training_log_category_{category}.csv', append=True)
        model_checkpoint = ModelCheckpoint(best_model, monitor='val_AUC', mode='max', save_best_only=True)

        class_weight = get_class_weights(train_df['Label'])
        print(f"class weights:{class_weight}")
        print(f"Training model for category '{category}'...")
        model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[early_stopping, csv_logger, model_checkpoint],
            class_weight=class_weight
        )
        # Save model
        model.save(model_save_path)
        print(f"Saved model for category '{category}' to {model_save_path}")
        category_models[category] = model
        print(f"Saved model {model} trying to load saved model")
        loaded_model = load(best_model)
        print(f"loaded model:{loaded_model}")
        del train_data, val_data, train_generator, val_generator, model
        import gc
        gc.collect()
    return category_models


def is_colab():
    return 'COLAB_GPU' in os.environ
def negative_train_test_split(train_df, y=None, test_size=0.2, random_state=42, stratify=None):
    if test_size > 0.0:
        if y is None:
            train_data, val_data = train_test_split(train_df, test_size=test_size, random_state=random_state, stratify=None)
        else:
            train_data, val_data, y_train, y_val = train_test_split(train_df, y, test_size=test_size, random_state=random_state, stratify=None)
    else:
        train_data=train_df
        val_data = pd.DataFrame([{}])
    if y is None:
        return train_data, val_data
    else:
        return train_data, val_data, y_train, y_val
def make_clustered_data(train_df, num_clusters, test_size=0.2, random_state=42, split_indepently=False):
    clustered_data = {}

    total_train_data, total_val_data = negative_train_test_split(train_df, test_size=test_size, random_state=random_state, stratify=None)
    for cluster in range(num_clusters):
        if split_indepently:
            cluster_data = train_df[train_df['Cluster'] == cluster]
            if cluster_data.empty:
                print(f"No data for Cluster {cluster}.")
                continue
            train_data, val_data = negative_train_test_split(cluster_data, test_size=test_size, random_state=random_state, stratify=None)
            clustered_data[cluster] = {
                'train': train_data.reset_index(drop=True),
                'val': val_data.reset_index(drop=True)
            }
        else:
            train_data = total_train_data[total_train_data['Cluster'] == cluster]
            val_data = total_val_data[total_val_data['Cluster'] == cluster] if not total_val_data.empty else pd.DataFrame([{}])
            if train_data.empty:
                print(f"No data for Cluster {cluster}.")
                continue
            clustered_data[cluster] = {
                'train': train_data.reset_index(drop=True)
            }
            clustered_data[cluster]['val'] = val_data.reset_index(drop=True)
        print(f"Cluster {cluster}: {len(train_data)} training samples, {len(val_data)} validation samples.")
    return clustered_data

def make_category_data(train_df, test_size=0.2, random_state=42, split_indepently=False):
    # Returns a dict like {'Sports': {'train': …, 'val': …}, …}
    category_data = {}
    if split_indepently:
        for cat, cat_df in train_df.groupby("CandidateCategory"):
            print(f"cat:{cat}, cat_df:{cat_df}")
            if cat_df.empty:
                continue
            tr, val = negative_train_test_split(cat_df, test_size=test_size, random_state=random_state, stratify=None)
            category_data[cat] = {
                "train": tr.reset_index(drop=True),
                "val":   val.reset_index(drop=True)
            }
            print(f"[{cat}] {len(tr)} train / {len(val)} val samples")
    else:
        tr, val = negative_train_test_split(train_df, test_size=test_size, random_state=random_state, stratify=None)
        for cat, cat_df in tr.groupby("CandidateCategory"):
            print(f"cat:{cat}, cat_df:{cat_df}")
            if cat_df.empty:
                continue
            category_data[cat] = {
                "train": tr.reset_index(drop=True)
            }
        if val.empty:
            log_print("No validation data !!!!!!!!!!!!")
            category_data[cat]["val"] = val.reset_index(drop=True)
        else:
            for cat, cat_df in val.groupby("CandidateCategory"):
                print(f"cat:{cat}, cat_df:{cat_df}")
                if cat_df.empty:
                    continue
                category_data[cat]["val"] = val.reset_index(drop=True)
    return category_data

def make_user_cluster_df(user_category_profiles_path = 'user_category_profiles.pkl', user_cluster_df_path = 'user_cluster_df.pkl'):
    user_category_profiles = pd.read_pickle(user_category_profiles_path)
    scaler = StandardScaler()
    user_profiles_scaled = scaler.fit_transform(user_category_profiles)

    scaler_path = 'user_profiles_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    # Clustering into 3 clusters based on KMeans
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(user_profiles_scaled)

    clustering_model_path = 'kmeans_user_clusters.pkl'
    with open(clustering_model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"Saved KMeans clustering model to {clustering_model_path}")

    user_clusters = kmeans.predict(user_profiles_scaled)
    user_category_profiles['Cluster'] = user_clusters
    user_cluster_df = user_category_profiles[['Cluster']]
    user_cluster_df.to_pickle(user_cluster_df_path)
    print(f"Saved user cluster assignments to {user_cluster_df_path}")

def load_dataset(data_dir, news_file='news.tsv', behaviors_file='behaviors.tsv'):
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(news_path, sep='\t',
                          names=['NewsID','Category','SubCategory','Title','Abstract','URL','TitleEntities','AbstractEntities'],
                          index_col=False)
    print(f"Loaded news data from {news_path}: {news_df.shape}")

    behaviors_path = os.path.join(data_dir, behaviors_file)
    behaviors_df = pd.read_csv(behaviors_path, sep='\t',
                               names=['ImpressionID','UserID','Time','HistoryText','Impressions'],
                               index_col=False)
    print(f"Loaded behaviors data from {behaviors_path}: {behaviors_df.shape}")
    behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna('')
    return news_df, behaviors_df

def get_midpoint_time(behaviors_df):
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    log_print(f"Computing Midpoint time for behaviors_df: {behaviors_df}")
    min_time = behaviors_df['Time'].min()
    max_time = behaviors_df['Time'].max()
    midpoint = min_time + (max_time - min_time) / 2
    log_print(f"min_time time: {min_time}")
    log_print(f"max_time time: {max_time}")
    log_print(f"Midpoint time computed: {midpoint}")
    return midpoint

def init_dataset(data_dir, news_file='news.tsv', behaviors_file='behaviors.tsv'):
    news_df, behaviors_df = load_dataset(data_dir, news_file, behaviors_file)
    # Create a combined text field for news
    news_df['CleanTitle'] = news_df['Title'].apply(clean_text)
    news_df['CleanAbstract'] = news_df['Abstract'].apply(clean_text)
    news_df['CombinedText'] = news_df['CleanTitle'] + ' ' + news_df['CleanAbstract']
    news_df["CombinedText"] = news_df["CombinedText"].astype(str).fillna("")
    return news_df, behaviors_df

def prepare_tokenizer(news_df, max_title_length=30):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")
    # Save tokenizer for later use
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    return tokenizer, vocab_size

def unzip_datasets(data_dir, valid_data_dir, zip_file, valid_zip_file):
    zip_file_path = f"{data_dir}{zip_file}"
    valid_zip_file_path = f"{valid_data_dir}{valid_zip_file}"
    local_file_path = os.path.join(data_dir, zip_file)
    local_valid_file_path = os.path.join(valid_data_dir, valid_zip_file)
    if not os.path.exists(local_file_path):
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename=zip_file,
            local_dir=data_dir
        )
    if not os.path.exists(local_valid_file_path):
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename=valid_zip_file,
            local_dir=valid_data_dir
        )
    output_folder = os.path.dirname(zip_file_path)
    valid_output_folder = os.path.dirname(valid_zip_file_path)
    
    # Unzip the file
    print(f"unzip {local_file_path}")
    with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    print(f"unzip {local_valid_file_path}")
    with zipfile.ZipFile(local_valid_file_path, 'r') as zip_ref:
        zip_ref.extractall(valid_output_folder)
    if is_colab():
      valid_output_folder = os.path.dirname(valid_zip_file_path)
      with zipfile.ZipFile(valid_zip_file_path, 'r') as zip_ref:
          zip_ref.extractall(os.path.dirname(valid_output_folder))
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'

def init(process_dfs = False, process_behaviors = False, data_dir = 'dataset/train/', train_data_dir = 'dataset/train/', valid_data_dir = 'dataset/valid/', zip_file = f"MINDlarge_train.zip", valid_zip_file = f"MINDlarge_dev.zip", download=False,
    news_df_pkl="models/news_df_processed.pkl", train_df_pkl="models/train_df_processed.pkl", categorized_samples=False, test_size=0.2, dataset="train", process_valid_sets=False, dataset_size="small",
    behavior_pickles=[[]], eval_frac=1.0, big_tokenizer=False, pivots=[45], ks=[5]):
    print(f"train_df_pkl={train_df_pkl},news_df_pkl={news_df_pkl}")
    global vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters
    if is_colab():
        print("Running on Google colab")
        data_dir = '/content/train/'
        valid_data_dir = '/content/valid/'
    #data_dir = 'dataset/small/train/'  # Adjust path as necessary
    #zip_file = f"MINDsmall_train.zip"
    unzip_datasets(data_dir, valid_data_dir, zip_file, valid_zip_file)
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'
    
    if dataset == "valid":
        data_dir = valid_data_dir

    # Load news data
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    
    print("Loaded news data:")
    print(news_df.head())
    
    # Load behaviors data
    behaviors_path = os.path.join(data_dir, behaviors_file)
        
    modified_behaviors_df_full = None
    modified_behaviors_df_k10 = None
    modified_behaviors_df_swap = None
    #if dataset == "valid":
    #    modified_behaviors_df_full = pd.read_pickle(f"evaluation_{dataset}_{dataset_size}_full.pkl")
    #    modified_behaviors_df_k10 = pd.read_pickle(f"evaluation_{dataset}_{dataset_size}_k10.pkl")
    #    modified_behaviors_df_swap = pd.read_pickle(f"evaluation_{dataset}_{dataset_size}_swap.pkl")
    
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    print("\nLoaded behaviors data:")
    print(behaviors_df.head())

    #print("\nLoaded modified_behaviors_df_full data:")
    #print(modified_behaviors_df_full.head())

    valid_news_path = os.path.join(valid_data_dir, news_file)
    valid_news_df = pd.read_csv(
        valid_news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    
    print("Loaded news data:")
    print(valid_news_df.head())
    
    # Load behaviors data
    valid_behaviors_path = os.path.join(valid_data_dir, behaviors_file)
    valid_behaviors_df = pd.read_csv(
        valid_behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    
    print("\nLoaded behaviors data:")
    print(valid_behaviors_df.head())
    if process_behaviors:
        # Handle missing 'HistoryText' by replacing NaN with empty string
        behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna('')
        
        # Create a NewsID to Category mapping
        newsid_to_category = news_df.set_index('NewsID')['Category'].to_dict()
        
        # Function to extract categories from HistoryText
        def extract_categories(history_text):
            if not history_text:
                return []
            news_ids = history_text.split(' ')
            categories = [newsid_to_category.get(news_id, 'Unknown') for news_id in news_ids]
            return categories
        
        # Apply the function to extract categories
        behaviors_df['HistoryCategories'] = behaviors_df['HistoryText'].apply(extract_categories)
        
        print("\nSample HistoryCategories:")
        print(behaviors_df[['UserID', 'HistoryCategories']].head())
        from collections import defaultdict
        
        # Initialize a dictionary to hold category counts per user
        user_category_counts = defaultdict(lambda: defaultdict(int))
        
        # Populate the dictionary
        for idx, row in behaviors_df.iterrows():
            user_id = row['UserID']
            categories = row['HistoryCategories']
            for category in categories:
                user_category_counts[user_id][category] += 1
        
        # Convert the dictionary to a DataFrame
        user_category_profiles = pd.DataFrame(user_category_counts).T.fillna(0)
        
        # Optionally, rename columns to indicate category
        user_category_profiles.columns = [f'Category_{cat}' for cat in user_category_profiles.columns]
        
        print("\nCreated user_category_profiles:")
        print(user_category_profiles.head())
        print(f"\nShape of user_category_profiles: {user_category_profiles.shape}")
        # Handle missing 'HistoryText' by replacing NaN with empty string
        behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna('')
        
        # Create a NewsID to Category mapping
        newsid_to_category = news_df.set_index('NewsID')['Category'].to_dict()
        
        # Get all unique UserIDs from behaviors_df
        unique_user_ids = behaviors_df['UserID'].unique()
        
        # Function to extract categories from HistoryText
        def extract_categories(history_text):
            if not history_text:
                return []
            news_ids = history_text.split(' ')
            categories = [newsid_to_category.get(news_id, 'Unknown') for news_id in news_ids]
            return categories
        
        # Apply the function to extract categories
        behaviors_df['HistoryCategories'] = behaviors_df['HistoryText'].apply(extract_categories)
        
        # Explode 'HistoryCategories' to have one category per row
        behaviors_exploded = behaviors_df[['UserID', 'HistoryCategories']].explode('HistoryCategories')
        
        # Replace missing categories with 'Unknown'
        behaviors_exploded['HistoryCategories'] = behaviors_exploded['HistoryCategories'].fillna('Unknown')
        
        # Create a cross-tabulation (pivot table) of counts
        user_category_counts = pd.crosstab(
            index=behaviors_exploded['UserID'],
            columns=behaviors_exploded['HistoryCategories']
        )
        
        # Rename columns to include 'Category_' prefix
        user_category_counts.columns = [f'Category_{col}' for col in user_category_counts.columns]
        
        # Reindex to include all users, filling missing values with zero
        user_category_profiles = user_category_counts.reindex(unique_user_ids, fill_value=0)
        
        print(f"\nCreated user_category_profiles with {user_category_profiles.shape[0]} users and {user_category_profiles.shape[1]} categories.")
        
        # Determine top N categories
        top_n = 20
        category_counts = news_df['Category'].value_counts()
        top_categories = category_counts.nlargest(top_n).index.tolist()
        
        # Get the category names without the 'Category_' prefix
        user_category_columns = user_category_profiles.columns.str.replace('Category_', '')
        
        # Filter columns in user_category_profiles that are in top_categories
        filtered_columns = user_category_profiles.columns[user_category_columns.isin(top_categories)]
        
        # Create filtered_user_category_profiles with these columns
        filtered_user_category_profiles = user_category_profiles[filtered_columns]
        
        # Identify columns that are not in top_categories to sum them into 'Category_Other'
        other_columns = user_category_profiles.columns[~user_category_columns.isin(top_categories)]
        
        # Sum the 'Other' categories
        filtered_user_category_profiles['Category_Other'] = user_category_profiles[other_columns].sum(axis=1)
        
        # Now, get the actual categories present after filtering
        actual_categories = filtered_columns.str.replace('Category_', '').tolist()
        
        # Add 'Other' to the list
        actual_categories.append('Other')
        print(f"Number of new column names: {len(actual_categories)}")
        # Assign new column names
        filtered_user_category_profiles.columns = [f'Category_{cat}' for cat in actual_categories]
        print("\nFiltered user_category_profiles with Top N Categories and 'Other':")
        print(filtered_user_category_profiles.head())
        print(f"\nShape of filtered_user_category_profiles: {filtered_user_category_profiles.shape}")
        
        # Save the user_category_profiles to a file for future use
        user_category_profiles_path = f"{dataset}_{dataset_size}_user_category_profiles.pkl"
        behaviors_df_processed_path = f"{dataset}_{dataset_size}_behaviors_df_processed.pkl"
        
        filtered_user_category_profiles.to_pickle(user_category_profiles_path)
        user_category_profiles = filtered_user_category_profiles
        print(f"\nSaved user_category_profiles to {user_category_profiles_path}")
        behaviors_df.to_pickle(behaviors_df_processed_path)
        print(f"\nSaved behaviors_df to {behaviors_df_processed_path}")
    else:
        """
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename="user_category_profiles.pkl",
            local_dir="models"
        )
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename="behaviors_df_processed.pkl",
            local_dir="models"
        )
        """
        user_category_profiles_path = f"{dataset}_{dataset_size}_user_category_profiles.pkl"
        behaviors_df_processed_path = f"{dataset}_{dataset_size}_behaviors_df_processed.pkl"
        user_category_profiles = pd.read_pickle(user_category_profiles_path)
        behaviors_df = pd.read_pickle(behaviors_df_processed_path)

    print(f"Number of columns in user_category_profiles: {len(user_category_profiles.columns)}")
    # Number of unique users in behaviors_df
    unique_user_ids = behaviors_df['UserID'].unique()
    print(f"Number of unique users in behaviors_df: {len(unique_user_ids)}")
    # Number of unique users in behaviors_df
    unique_user_ids = behaviors_df['UserID'].unique()
    print(f"Number of unique users in behaviors_df: {len(unique_user_ids)}")
    
    # Number of users in user_category_profiles
    user_profile_ids = user_category_profiles.index.unique()
    print(f"Number of users in user_category_profiles: {len(user_profile_ids)}")
    
    # Find missing UserIDs
    missing_user_ids = set(unique_user_ids) - set(user_profile_ids)
    print(f"Number of missing UserIDs in user_category_profiles: {len(missing_user_ids)}")
    tokenizer = Tokenizer()
    num_clusters = 3
    if process_dfs:
        # processing eval sets next
        # train_df_pkl_full=None, news_df_pkl_full=None, train_df_pkl_k10=None, news_df_pkl_k10=None, train_df_pkl_swap=None, news_df_pkl_swap=None,
        """
        for behavior_pickle_l in behavior_pickles:
            behavior_pickle, train_df_pkl_full, news_df_pkl_full = behavior_pickle_l
            clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
                data_dir=data_dir,
                news_file=news_file,
                behaviors_file=behaviors_file,
                user_category_profiles=user_category_profiles,
                num_clusters=num_clusters,
                fraction=1,
                max_title_length=30,
                max_history_length=50,
                train_df_pkl=train_df_pkl_full, news_df_pkl=news_df_pkl_full, categorized_samples=categorized_samples,
                test_size=test_size,
                dataset=dataset, process_valid_sets=process_valid_sets, dataset_size=dataset_size, behavior_pickle=behavior_pickle
            )
        """
        ###
        clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
            data_dir=data_dir,
            train_data_dir=train_data_dir,
            valid_data_dir=valid_data_dir,
            news_file=news_file,
            behaviors_file=behaviors_file,
            user_category_profiles=user_category_profiles,
            num_clusters=num_clusters,
            fraction=1,
            max_title_length=30,
            max_history_length=50,
            train_df_pkl=train_df_pkl, news_df_pkl=news_df_pkl, categorized_samples=categorized_samples,
            test_size=test_size,
            dataset=dataset, process_valid_sets=process_valid_sets, dataset_size=dataset_size,
            eval_frac=eval_frac, big_tokenizer=big_tokenizer, pivots=pivots, ks=ks
        )

    ###
    if download:
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename="news_df_processed.pkl",
            local_dir="models"
        )
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename="train_df_processed.pkl",
            local_dir="models"
        )
    news_df = pd.read_pickle(news_df_pkl)
    train_df = pd.read_pickle(train_df_pkl)

    clustered_data = make_clustered_data(train_df, num_clusters,test_size=test_size)
    if categorized_samples:
        category_data = make_category_data(train_df,test_size=test_size)
    else:
        category_data = {}

    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    max_history_length = 50
    max_title_length = 30
    batch_size = 64
    return data_dir, vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters, category_data, modified_behaviors_df_full, modified_behaviors_df_k10, modified_behaviors_df_swap
def quick_compare(bdf, tdf, n=5):
    both = pd.concat([bdf.head(n).assign(src="behaviors"),
                      tdf.head(n).assign(src="train")])
    print("Shapes  to", bdf.shape, tdf.shape)
    print("Columns to", set(bdf.columns) ^ set(tdf.columns))  # symmetric diff
    print(both.sort_index())


def get_class_weights(labels):
    print(f"get_class_weights: labels={labels}")
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

def load_category_models(category_train_dfs, dataset_size='large', model_size='large', epochs=3, load_best_model=True, tokenizer=None,
    max_history_length=50, max_title_length=30, train_new=False, batch_size=256, model_base=''):
    model_dir = "models"
    category_models = {}
    for category, df in category_train_dfs.items():
        if model_base == '':
            model_file_prefix=f"fastformer_{model_size}_category_{category}_{epochs}epochs"
        else:
            model_file_prefix = f'fastformer_category_{category}_{model_base}'
        keras_model_save_path = f'{model_dir}/{model_file_prefix}.keras'
        model_save_path = f'{model_dir}/{model_file_prefix}.h5'
        best_model = f'{model_dir}/best_model_{model_file_prefix}.h5'
        keras_best_model = f'{model_dir}/{model_file_prefix}.keras'
        if load_best_model and os.path.exists(best_model):
            category_models[f"{category}_{model_size}"] = load(best_model)
        elif not load_best_model and os.path.exists(model_save_path):
            category_models[f"{category}_{model_size}"] = load(model_save_path)
        
    if category_models == {} or train_new:
        vocab_size = len(tokenizer.word_index) + 1
        category_models = train_category_models(category_train_dfs, vocab_size, max_history_length, max_title_length, batch_size=batch_size, epochs=epochs, dataset_size=dataset_size,
            train_only_new=False, train_fraction=0.9, load_best_model=True, model_base=model_base)
    return category_models

def log_print(s):
    logging.info(s)
    print(s)

def print_to_file(text: str, filename: str, mode: str = 'a', encoding: str = 'utf-8'):
    with open(filename, mode, encoding=encoding) as f:
        print(text, file=f)

def add_candidate_tokens(df, news_text_dict, max_title_len=30):
    df = df.copy()
    log_print(f"creawting CandidateTitleTokens column")
    df["CandidateTitleTokens"] = df["CandidateID"].map(
        lambda nid: news_text_dict.get(nid, [0]*max_title_len)
    )
    log_print(f"added column df['CandidateTitleTokens']: {df['CandidateTitleTokens']}")
    return df

def _load_eval(path):
    log_print(f"loading {path}")
    df = pd.read_pickle(path)
    #df = pd.read_parquet(path, engine="pyarrow", dtype_backend="pyarrow")
    #for col in ("HistoryTitles", "CandidateTitleTokens"):
    #    df[col] = df[col].tolist()
    """
    df = pd.read_parquet(
        path,
        engine="pyarrow",
        dtype_backend="numpy_nullable",
    )
    
    df = pd.read_parquet(
            path,
            engine="pyarrow",
            dtype_backend="pyarrow"
    )
    .astype({
        "HistoryTitles":        "object",
        "CandidateTitleTokens": "object",
    })
    log_print(f"finished loading {path}")
    df["HistoryTitles"] = df["HistoryTitles"].apply(
        lambda seqs: [[int(tok) for tok in title] for title in seqs]
    )
    df["CandidateTitleTokens"] = df["CandidateTitleTokens"].apply(
        lambda toks: [int(tok) for tok in toks]
    )
    log_print(f"finished loading {path}")
    for col in ("HistoryTitles", "CandidateTitleTokens"):
        df[col] = df[col].array.to_numpy(zero_copy_only=True)
    df = df.astype({
            "HistoryTitles":        "object",
            "CandidateTitleTokens": "object",
    })
    
    log_print(f"finished loading {path}")
    for col in ("HistoryTitles", "CandidateTitleTokens"):
        df[col] = (df[col]
            .to_pylist()
            .apply(lambda x: np.asarray(x, dtype="int32")))
    log_print(f"finished convert")
    """
    return df

def get_models(process_dfs = False, process_behaviors = False, data_dir = 'dataset/train/', valid_data_dir = 'dataset/valid/', zip_file = f"MINDlarge_train.zip", valid_zip_file = f"MINDlarge_dev.zip",
    model_type='cluster', dataset_size='large', model_size='large', load_best_model=False, load_best_models=[], epochs=1, retrain_models = [], evaluate=False, skip_already_evaluated=False, dataset_fraction=1.0,
    dataset='train', batch_size=256, eval_dataset_size='large', cutoff_time_str=None, eval_full=False, test_size=0.0, use_model_base=False, process_valid_sets=False, end_after_preprocess=False, make_title_tensor=False,
    eval_frac=1.0, big_tokenizer=False, small_train_data_dir = "dataset/small/train/", pivots=[45], ks=[5], reverse=False, tune_threshold=False, regenerate_metrics=False, vers_suffix="_v4", model_arc_type="fastformer",
    timed=False, use_cpu=False, clear_store=False):
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'

    small_news_df_pkl = "models/small_news_df_processed"
    small_train_df_pkl = "models/small_train_df_processed"
    news_df_pkl = "models/news_df_processed"
    train_df_pkl = "models/train_df_processed"
    categorized_samples = False
    if model_type == 'category' or model_type == 'all':
        small_news_df_pkl = f"{small_news_df_pkl}_categorized"
        small_train_df_pkl = f"{small_train_df_pkl}_categorized"
        news_df_pkl = f"{news_df_pkl}_categorized"
        train_df_pkl = f"{train_df_pkl}_categorized"
        categorized_samples = True
    if dataset != 'train':
        small_news_df_pkl = f"{small_news_df_pkl}_{dataset}"
        small_train_df_pkl = f"{small_train_df_pkl}_{dataset}"
        news_df_pkl = f"{news_df_pkl}_{dataset}"
        train_df_pkl = f"{train_df_pkl}_{dataset}"

    small_news_df_pkl_full = f"{small_news_df_pkl}_full.pkl"
    small_train_df_pkl_full = f"{small_train_df_pkl}_full.pkl"
    news_df_pkl_full = f"{news_df_pkl}_full.pkl"
    train_df_pkl_full = f"{train_df_pkl}_full.pkl"
    small_news_df_pkl_k10 = f"{small_news_df_pkl}_k10.pkl"
    small_train_df_pkl_k10 = f"{small_train_df_pkl}_k10.pkl"
    news_df_pkl_k10 = f"{news_df_pkl}_k10.pkl"
    train_df_pkl_k10 = f"{train_df_pkl}_k10.pkl"
    small_news_df_pkl_swap = f"{small_news_df_pkl}_swap.pkl"
    small_train_df_pkl_swap = f"{small_train_df_pkl}_swap.pkl"
    news_df_pkl_swap = f"{news_df_pkl}_swap.pkl"
    train_df_pkl_swap = f"{train_df_pkl}_swap.pkl"


    small_news_df_pkl = f"{small_news_df_pkl}.pkl"
    small_train_df_pkl = f"{small_train_df_pkl}.pkl"
    if "small" in data_dir:
        news_df_pkl = small_news_df_pkl
        train_df_pkl = small_train_df_pkl
        news_df_pkl_full = small_news_df_pkl_full
        train_df_pkl_full = small_train_df_pkl_full
        news_df_pkl_k10 = small_news_df_pkl_k10
        train_df_pkl_k10 = small_train_df_pkl_k10
        news_df_pkl_swap = small_news_df_pkl_swap
        train_df_pkl_swap = small_train_df_pkl_swap
    else:
        news_df_pkl = f"{news_df_pkl}.pkl"
        train_df_pkl = f"{train_df_pkl}.pkl"
        news_df_pkl_full = news_df_pkl_full
        train_df_pkl_full = train_df_pkl_full
        news_df_pkl_k10 = news_df_pkl_k10
        train_df_pkl_k10 = train_df_pkl_k10
        news_df_pkl_swap = news_df_pkl_swap
        train_df_pkl_swap = train_df_pkl_swap

    data_dir, vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters, category_data, modified_behaviors_df_full, modified_behaviors_df_k10, modified_behaviors_df_swap = init(process_dfs, process_behaviors, data_dir, small_train_data_dir, valid_data_dir, zip_file, valid_zip_file,
    train_df_pkl=train_df_pkl, news_df_pkl=news_df_pkl, categorized_samples=categorized_samples, test_size=test_size, dataset=dataset, process_valid_sets=process_valid_sets,dataset_size=dataset_size,
    behavior_pickles=[[f"evaluation_{dataset}_{dataset_size}_full.parquet",train_df_pkl_full,news_df_pkl_full],
    [f"evaluation_{dataset}_{dataset_size}_k10.parquet",train_df_pkl_k10,news_df_pkl_k10],
    [f"evaluation_{dataset}_{dataset_size}_swap.parquet",train_df_pkl_swap,news_df_pkl_swap]], eval_frac=eval_frac, big_tokenizer=big_tokenizer, pivots=pivots, ks=ks)
    """
    if process_dfs:
        clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
            data_dir=data_dir,
            valid_data_dir=valid_data_dir,
            news_file=news_file,
            behaviors_file=behaviors_file,
            user_category_profiles=user_category_profiles,
            num_clusters=3,
            fraction=1,
            max_title_length=30,
            max_history_length=50,
            train_df_pkl=train_df_pkl, news_df_pkl=news_df_pkl, categorized_samples=categorized_samples,
            test_size=test_size,
            dataset=dataset,process_valid_sets=process_valid_sets,dataset_size=dataset_size,
            eval_frac=eval_frac
        )
    """
    if not os.path.exists(train_df_pkl) or not os.path.exists(news_df_pkl):
        log_print(f"\n !!! train_df_pkl: {train_df_pkl} DOESN'T EXIST switching to use small_train_df_pkl:{small_train_df_pkl} or news_df_pkl: {news_df_pkl} DOESN'T EXIST switching to use small_news_df_pkl:{small_news_df_pkl} !!!!\n !!! train_df_pkl: {train_df_pkl} DOESN'T EXIST switching to use small_train_df_pkl:{small_train_df_pkl} !!!!\n !!! train_df_pkl: {train_df_pkl} DOESN'T EXIST !!!!\n !!! train_df_pkl: {train_df_pkl} DOESN'T EXIST !!!!\n !!! train_df_pkl: {train_df_pkl} DOESN'T EXIST !!!!")
        news_df_pkl = small_news_df_pkl
        train_df_pkl = small_train_df_pkl
    if test_size > 0.0:
        train_data, test_data = negative_train_test_split(train_df, test_size=test_size, random_state=42, stratify=None)
        #category_data = make_category_data(train_df, test_size=test_size, random_state=42)
    else:
        train_data = train_df
        test_data = train_df
    quick_compare(behaviors_df, train_df)
    model_base = ''
    print(f"model_base:{model_base}")
    if use_model_base:
        model_base = f"{model_size}_full_balanced_{epochs}_epoch"
        if test_size > 0.0:
            model_base = f"{model_size}_train_size_{1-test_size}_balanced_{epochs}_epoch"
        if big_tokenizer:
            model_base = f"{model_base}_big_tokenizer"
        if timed:
            model_base = f"{model_base}_timed"

    print(f"model_base:{model_base}")

    if model_type == 'cluster' or model_type == 'all':
        cluster_models = train_cluster_models(
            clustered_data=clustered_data,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            max_history_length=max_history_length,
            max_title_length=max_title_length,
            num_clusters=num_clusters,
            batch_size=batch_size,
            epochs=epochs,
            size=model_size,
            retrain='cluster' in retrain_models,
            model_base=model_base,
            dataset_size=dataset_size,
            dataset=dataset,
            test_size=test_size
        )
        logging.info(f"loaded total cluster_models:{cluster_models}")
    if model_type == 'global' or model_type == 'all':
        global_model = train_global_model(train_data, tokenizer, vocab_size, max_history_length, max_title_length, val_data=test_data, dataset_size=model_size, batch_size=batch_size, epochs=epochs,
        retrain='global' in retrain_models, load_best_model='global' in load_best_models, model_base=model_base, model_arc_type=model_arc_type, timed=timed)
        global_models = {}
        global_models[f"global_{model_size}"] = global_model
        logging.info(f"loaded total global_models:{global_models}")
    if model_type == 'category' or model_type == 'all':
        #category_train_dfs, news_df, behaviors_df, tokenizer = prepare_category_train_dfs(data_dir, news_file, behaviors_file, 30, 50, f"category_train_dfs_{dataset_size}.pkl")
        category_train_dfs = train_df
        category_models = load_category_models(category_data, dataset_size=dataset_size, model_size=model_size, load_best_model=load_best_model or 'category' in load_best_models,
        epochs=epochs, tokenizer=tokenizer, train_new='category' in retrain_models, batch_size=batch_size, model_base=model_base)
    if model_type == "cluster":
        models = cluster_models
    elif model_type == "category":
        models = category_models
    elif model_type == "global":
        models = global_models
    elif model_type == "all":
        models = {**cluster_models, **category_models, **global_models}

    logging.info(f"loaded total models:{models}")
    logging.info(f"difference between train_df:{train_df} and behaviors_df:{behaviors_df}")
    if evaluate:
        out_dir = Path("base_preds")
        out_dir.mkdir(exist_ok=True)
        if not eval_full:
            if test_size > 0.0:
                train_data, test_data = negative_train_test_split(train_df, test_size=test_size, random_state=42, stratify=None)
            #train_data, test_data = negative_train_test_split_time(train_df, cutoff_time_str)
        else:
            test_size=1.0
            test_data = train_df
        
        title_tensor, id_to_index = get_title_tensors(dataset, dataset_size, news_df, tokenizer, make_new=make_title_tensor)
        subset_tag = "" if eval_frac >= 1.0 else f"_{eval_frac}"
        col_dfs = []
        ## SETS DATASET AS VALID LARGE WHEN TRIAN DATASET
        if dataset == 'train':
            dataset = 'valid'
            dataset_size = 'large'
        col = f"Hist_full{subset_tag}"
        col_dfs.append([col, _load_eval(f"evaluation_{dataset}_{dataset_size}{subset_tag}_{col}.pkl")])
        for pivot in pivots:
            col = f"Hist_swap{pivot}{subset_tag}"
            col_dfs.append([col, _load_eval(f"evaluation_{dataset}_{dataset_size}{subset_tag}_{col}.pkl")])
        for k in ks:
            col = f"Hist_k{k}{subset_tag}"
            col_dfs.append([col, _load_eval(f"evaluation_{dataset}_{dataset_size}{subset_tag}_{col}.pkl")])
        items = list(models.items())
        if reverse:
            items.reverse()
            print(f"reversed:{items}")
        if timed:
            vers_suffix = f"{vers_suffix}_timed"
            tf.config.run_functions_eagerly(True)
            tf.config.experimental.set_synchronous_execution(True)
        if use_cpu:
            vers_suffix = f"{vers_suffix}_cpu"
        for key, model in items:
            key = f"{model_arc_type}_{key}"
            base_store_file = f"{key}_{dataset}_{dataset_size}_test_size_{test_size}{vers_suffix}"
            base_store_path = out_dir / f"{base_store_file}"
            base_store_metrics_file = f"{key}_{dataset}_{dataset_size}_test_size_{test_size}_metrics{vers_suffix}"
            base_store_metrics_path = out_dir / f"0.5000_{base_store_metrics_file}"
            store_path = out_dir / f"{key}_{dataset}_{dataset_size}_test_size_{test_size}{vers_suffix}.parquet"
            store_metrics_path = out_dir / f"{key}_{dataset}_{dataset_size}_test_size_{test_size}_metrics{vers_suffix}.json"

            #print(f"loaded df_full:{df_full.head()}")
            #df_full.to_pickle(f"evaluation_{dataset}_{dataset_size}_full.pkl")
            #df_k10.to_pickle(f"evaluation_{dataset}_{dataset_size}_k10.pkl")
            #df_swap.to_pickle(f"evaluation_{dataset}_{dataset_size}_swap.pkl")
            #modified_behaviors_df_full = pd.read_pickle(f"evaluation_{dataset}_{dataset_size}_full.pkl")
            #modified_behaviors_df_k10 = pd.read_pickle(f"evaluation_{dataset}_{dataset_size}_k10.pkl")


            """
            if skip_already_evaluated and store_path.exists() and store_path.stat().st_size > 0:
                log_print(f"Skipping evaluation for {key}: {store_path}")
                continue
            log_print(f"evaluation with key:{key}, model:{model}, {store_path}:{store_path}, store_metrics_path:{store_metrics_path}")
            res = evaluate_with_generator(
                model,
                eval_df=test_data,
                batch_size=batch_size,
                store_path=store_path,
                flush_every=0.01,
                verbose=1,
                store_metrics_path=store_metrics_path
            )
            logging.info(f"model {key} eval res:{res}")
            log_print("SKIPS modified sets EVAL!!!")
            continue
            """


            for col, df in col_dfs:
                print(f"eval col: {col}")
                adapt_store_path = f"{base_store_path}_{col}.parquet"
                adapt_store_metrics_path = f"{base_store_metrics_path}_{col}.json"
                adapt_store_metrics_file = f"{base_store_metrics_file}_{col}.json"
                if regenerate_metrics:
                    regenerate_metrics_from_parquet(adapt_store_path,
                                    decision_threshold = 0.5,
                                    tune_threshold = False,
                                    store_metrics_dir=out_dir,
                                    store_metrics_file=adapt_store_metrics_file)
                    
                    log_print(f"Regenerated {adapt_store_path}, continuing to next")
                    continue
                if skip_already_evaluated and Path(adapt_store_path).exists() and Path(adapt_store_path).stat().st_size > 0:
                    log_print(f"Skipping evaluation for {key}, col:{col}: {adapt_store_path}")
                    continue
                log_print(f"evaluting with evaluation_{dataset}_{dataset_size}{subset_tag}_{col}.pkl")
                if timed:
                    try:
                        os.remove('logs/timings.log')
                    except FileNotFoundError:
                        pass
                if clear_store:
                    os.remove(adapt_store_path)
                res = evaluate_with_generator(
                    model,
                    eval_df=df,
                    batch_size=batch_size,
                    store_path=adapt_store_path,
                    flush_every=0.01,
                    verbose=1,
                    store_metrics_dir=out_dir,
                    store_metrics_file=adapt_store_metrics_file,
                    tune_threshold=tune_threshold,
                    model_key=key,
                    timed=timed
                )

            log_print("SKIPS NORMAL EVAL!!! EVALUATES ONLY ON MODIFIED DATASETS!!!!!!!!!!!!")
            continue

            if skip_already_evaluated and store_path.exists() and store_path.stat().st_size > 0:
                log_print(f"Skipping evaluation for {key}: {store_path}")
                continue
            log_print(f"evaluation with key:{key}, model:{model}, {store_path}:{store_path}, store_metrics_path:{store_metrics_path}")
            if timed:
                try:
                    os.remove('logs/timings.log')
                except FileNotFoundError:
                    pass
            res = evaluate_with_generator(
                model,
                eval_df=test_data,
                batch_size=batch_size,
                store_path=store_path,
                flush_every=0.01,
                verbose=1,
                store_metrics_path=store_metrics_path
            )
            logging.info(f"model {key} eval res:{res}")

            #baseline   = run_eval(pd.read_pickle(f"evaluation_{dataset}_{dataset_size}_full.pkl"),  model, title_tensor, id_to_index)
            #k10_drop   = run_eval(pd.read_pickle(f"evaluation_{dataset}_{dataset_size}_k10.pkl"),   model, title_tensor, id_to_index)
            #swap_drift = run_eval(pd.read_pickle(f"evaluation_{dataset}_{dataset_size}_swap.pkl"),  model, title_tensor, id_to_index)
            #log_print(f"baseline:{baseline}")
            #log_print(f"k10_drop:{k10_drop}")
            #log_print(f"swap_drift:{swap_drift}")
            #print("Δ nDCG@10 when we truncate:", baseline["ndcg@10"] - k10_drop["ndcg@10"])
            #print("nDCG@10 under sudden drift (swap):", swap_drift["ndcg@10"])
    set_global_models(models)
    return models, news_df, train_df, tokenizer # todo check train_df or behaviors_df?????

# Train meta model using base models' preds saved as .parquet files
def train_meta_from_parquet(
        parquet_dir: str = "base_preds",
        pattern: str = "*.parquet",
        out_path: str = "meta_model.joblib",
        base_model_type = "all",
        test_size = 0.2,
        C = 1.0,
        random_state = 42,
        verbose = True,
        estimator = "lbfgs",
        fit_intercept = True,
        class_weight  = "balanced",
        meta_model_type = 'LogisticRegression',
        booster = 'default',
        tree_method = 'default',
        n_estimators = 300):

    files = sorted(Path(parquet_dir).glob(pattern))
    if base_model_type == "cluster":
        #keep = re.compile(r"^(?:\d+|global)_")
        keep = re.compile(r"^fastformer_(?:\d+)_")
        files = [fp for fp in files if keep.match(fp.stem)]
    if base_model_type == "category":
        remove = re.compile(r"^fastformer_(?:\d+)_")
        files = [fp for fp in files if not remove.match(fp.stem)]

    if not files:
        raise FileNotFoundError(f"No parquet files matching {pattern} in {parquet_dir}")

    merged = None
    feature_cols = []
    print(f"base files:{files}")
    # Preprocess base model preds for meta model training

    for fp in files:
        df = pd.read_parquet(fp)
        model_name = fp.stem
        col_name   = f"pred_{model_name}"

        if "y_pred" not in df.columns or "y_true" not in df.columns:
            raise ValueError(f"{fp} must contain y_true and y_pred")

        df = df.rename(columns={"y_pred": col_name})
        feature_cols.append(col_name)

        keep = ["row_id", "y_true", col_name]

        merged = df[keep] if merged is None else merged.merge(df[keep], on=["row_id", "y_true"])
    X = merged[feature_cols].values.astype("float32")
    y = merged["y_true"].values.astype("int8")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Build and Train meta model
    if meta_model_type == 'SGDClassifier':
        meta = SGDClassifier(
            penalty      ="l2",
            fit_intercept=fit_intercept,
            class_weight =class_weight,
            max_iter     =1000,
            random_state =random_state,
        )
    elif meta_model_type == 'LogisticRegressionCV':
        meta = LogisticRegressionCV(
            Cs=np.logspace(-3,3,10),
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            scoring="roc_auc",
            n_jobs=-1,
            cv=5,
            class_weight=class_weight
        )
    elif meta_model_type == 'CalibratedClassifierCV_pipe':
        base = LogisticRegressionCV(
            Cs=np.logspace(-3,3,10),
            penalty="elasticnet",
            l1_ratios=[0.2, 0.5, 0.8],
            solver="saga",
            class_weight=class_weight,
            max_iter=4000,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        )
        pipe = make_pipeline(StandardScaler(with_mean=False), base)
        meta = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    elif meta_model_type == 'XGBClassifier':
        import xgboost
        if booster == 'gblinear':
            meta = xgboost.XGBClassifier(
                booster='gblinear',
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            if tree_method == 'hist':
                meta = xgboost.XGBClassifier(tree_method=tree_method, device="cuda")
            else:
                meta = xgboost.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="auc",
                )
    elif meta_model_type == 'CalibratedClassifierCV':
        meta = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.2,
            solver="saga",
            max_iter=3000,
            C=1.0,
            class_weight=class_weight,
        )
        meta = CalibratedClassifierCV(meta, method="isotonic", cv=3)
    else:
        meta = LogisticRegression(
            penalty      ="l2",
            C            =C,
            fit_intercept=fit_intercept,
            class_weight =class_weight,
            max_iter     =4000,
            solver       ="lbfgs",
            random_state =random_state,
        )

    meta.fit(X_train, y_train)

    # Print metrics
    if verbose:
        prob_val = meta.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob_val := prob_val)
        p, r, f1, _ = precision_recall_fscore_support(
            y_val, (prob_val >= 0.5).astype(int), average="binary", zero_division=0)
        print(json.dumps({"AUC": auc, "precision": p, "recall": r, "F1": f1}, indent=2))


    joblib.dump(meta, out_path)
    if verbose:
        print(f"Meta model trained on {X.shape[0]:,} rows using {len(feature_cols)} base models and saved to {out_path}")
        logging.info(f"Meta model trained on {X.shape[0]:,} rows using {len(feature_cols)} base models and saved to {out_path}")

    return meta

def gpu_predict_in_batches(model, X, batch=200000):
    out = []
    for start in range(0, X.shape[0], batch):
        part = X[start:start+batch]
        out.append(model.predict_proba(part)[:, 1])
    return np.concatenate(out)

def _best_threshold(y_true, y_pred):
    p, r, t = precision_recall_curve(y_true, y_pred)
    f1 = 2 * p * r / (p + r + 1e-9)
    return t[np.argmax(f1)] if len(t) else 0.5

from functools import reduce

import time, json, numpy as np, pandas as pd
from pathlib import Path
from functools import reduce
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
import joblib
import time, pyarrow.dataset as ds
import pyarrow.parquet as pq

def fast_stack(files):
    """
    Read N parquet files that all share identical row_id order
    and return one merged DataFrame [row_id, y_true, ImpressionID, pred_*].
    Much faster than N-way joins.
    """
    # ---------- read first file (labels once) ----------
    print(f"files:{files}")
    first = files[0]
    tbl   = pq.read_table(first, columns=["row_id",
                                          "y_true",
                                          "ImpressionID",
                                          "y_pred"])
    df    = tbl.to_pandas()
    df.rename(columns={"y_pred": f"pred_{first.stem}"}, inplace=True)

    # ---------- pre-allocate numpy block for remaining preds ----------
    n_rows   = len(df)
    n_models = len(files) - 1
    pred_mat = np.empty((n_rows, n_models), dtype="float32")
    names    = []

    lengths = [(fp, pq.read_metadata(fp).num_rows) for fp in files]
    print(f"lengths:{lengths}")
    base_len = lengths[0][1]
    for fp, n in lengths:
        if n != base_len:
            print(f"⚠️  {fp} has {n:,} rows (expected {base_len:,})")

    for j, fp in enumerate(files[1:]):
        col = pq.read_table(fp, columns=["y_pred"]).column(0)
        #.to_pandas()
        #col = col[col.y_pred.notna()]
        col = col.to_numpy()

        print(f"fp:{fp}")
        pred_mat[:, j] = col
        names.append(f"pred_{fp.stem}")

    df[names] = pred_mat      # single vectorised insertion
    return df

def evaluate_meta_from_parquet(meta_path: str,
                               parquet_dir: str = "base_preds",
                               pattern: str = "*.parquet",
                               base_model_type: str = "all",
                               store_parquet_path: str = "preds.parquet",
                               store_metrics_path: str = "metrics.json",
                               verbose: bool = True,
                               booster='default',
                               batches: bool = True,
                               model_arc_type = "fastformer"):

    overall_start = time.perf_counter()                         # ← added
    if verbose:
        print(f"loading meta model from {meta_path}")
    meta = joblib.load(meta_path)
    print(f"Path(parquet_dir).glob(pattern):{Path(parquet_dir).glob(pattern)}")

    # ------------------------------------------------------------------ #
    # 1) locate parquet files                                            #
    # ------------------------------------------------------------------ #
    files = sorted(Path(parquet_dir).glob(pattern))
    if base_model_type == "cluster":
        files = [fp for fp in files if re.match(r"^fastformer_(?:\d+)_", fp.stem)]
    elif base_model_type == "category":
        files = [fp for fp in files if not re.match(r"^fastformer_(?:global|\d+)_", fp.stem)]
    
    # REMOVES na as it somehow was not included in meta training set!!
    #files = [fp for fp in files if not re.match(r"^fastformer_northamerica_", fp.stem)]
    # REMOVES na!!
    
    if not files:
        raise FileNotFoundError("No parquet files...")

    # ------------------------------------------------------------------ #
    # 2) fast load & stack (no Python joins)                             #
    # ------------------------------------------------------------------ #
    load_start = time.perf_counter()
    print(f"fast_stack")
    merged     = fast_stack(files)            # <── NEW
    load_time  = time.perf_counter() - load_start
    merge_time = 0.0                          # we skipped the join step
    print(f"Loaded & stacked {len(files)} files in {load_time:.3f}s")
    """
    # ------------------------------------------------------------------ #
    # 1) locate parquet files & read them                                #
    # ------------------------------------------------------------------ #
    files = sorted(Path(parquet_dir).glob(pattern))
    if base_model_type == "cluster":
        files = [fp for fp in files if re.match(r"^(?:\d+)_", fp.stem)]
    elif base_model_type == "category":
        files = [fp for fp in files if not re.match(r"^(?:\d+)_", fp.stem)]
    if not files:
        raise FileNotFoundError(f"No parquet files matching {pattern}")
    
    
    
    io_start  = time.perf_counter()

    def load_and_prepare(fp, first=False):
        print(f"loading file:{fp}")
        # ❶ grab labels & impression ids just once
        cols = ["row_id", "y_true", "ImpressionID", "y_pred"] if first else ["row_id", "y_pred"]
        tbl  = ds.dataset(fp).to_table(columns=cols)          # vectorised Arrow reader
        df   = tbl.to_pandas()
        df   = df.rename(columns={"y_pred": f"pred_{fp.stem}"})
        return df.set_index("row_id")


    dfs = [load_and_prepare(fp, first=(i == 0))               # ❷
        for i, fp in enumerate(files)]

    io_time = time.perf_counter() - io_start                  # ← new metric
    print(f"merging next")
    # ------------------------------------------------------------------
    # 2) join all frames on their integer index in a single pass
    #    (“row_id” is already the index, so join is hash-based & O(n))
    # ------------------------------------------------------------------
    merge_start = time.perf_counter()
    merged = reduce(lambda a, b: a.join(b, how="inner"), dfs) # fast Arrow→Pandas join
    merge_time  = time.perf_counter() - merge_start           # ← new metric
    """
    featcols    = [c for c in merged.columns if c.startswith("pred_")]
    
    # split once to NumPy – no additional copies afterwards
    y_true   = merged.pop("y_true").to_numpy("int8", copy=False)
    impr_ids = merged.pop("ImpressionID").to_numpy("int64", copy=False)
    X        = merged[featcols].to_numpy("float32", copy=False)
    print(f"pred next")
    # ------------------------------------------------------------------
    # 3) meta-model / bagging inference with timing
    # ------------------------------------------------------------------
    pred_start = time.perf_counter()
    if batches:
        y_pred = gpu_predict_in_batches(meta, X, batch=200_000)
    else:
        y_pred = meta.predict_proba(X)[:, 1]
    predict_time = time.perf_counter() - pred_start

    merged["y_pred"] = y_pred.astype("float32")
    print(f"pred end")

    """
    load_t0 = time.perf_counter()                               # ← added
    def _load(fp, first=False):
        print(f"loading file:{fp}")
        cols = ["row_id", "y_true", "y_pred", "ImpressionID"] if first else ["row_id", "y_pred"]
        df   = pd.read_parquet(fp, columns=cols)
        return df.rename(columns={"y_pred": f"pred_{fp.stem}"}).set_index("row_id")
    dfs = [_load(fp, first=(i == 0)) for i, fp in enumerate(files)]
    load_time = time.perf_counter() - load_t0                   # ← added

    # ------------------------------------------------------------------ #
    # 2) merge all base-model predictions                                #
    # ------------------------------------------------------------------ #
    merge_t0 = time.perf_counter()                              # ← added
    merged = reduce(lambda l, r: l.join(r, how="inner"), dfs)
    merge_time = time.perf_counter() - merge_t0                 # ← added

    y_true   = merged.pop("y_true").to_numpy("int8")
    impr_ids = merged.pop("ImpressionID").to_numpy("int64")
    featcols = merged.columns.tolist()
    X        = merged.to_numpy("float32")

    # ------------------------------------------------------------------ #
    # 3) meta-model inference                                            #
    # ------------------------------------------------------------------ #
    pred_t0 = time.perf_counter()                               # ← added
    if batches:
        y_pred = gpu_predict_in_batches(meta, X, batch=200_000)
    else:
        y_pred = meta.predict_proba(X)[:, 1]
    predict_time = time.perf_counter() - pred_t0                # ← added

    # attach predictions
    merged["y_pred"] = y_pred.astype("float32")
    """
    # ------------------------------------------------------------------ #
    # 4) compute meta-model metrics                                      #
    # ------------------------------------------------------------------ #
    auc  = roc_auc_score(y_true, y_pred)
    ap   = average_precision_score(y_true, y_pred)
    thr  = _best_threshold(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, (y_pred >= thr).astype(int), average="binary", zero_division=0)
    n_samples = len(y_true)
    metrics = {
        "samples": len(y_true),
        "AUC":      float(auc),
        "AP":       float(ap),
        "precision":float(p),
        "recall":   float(r),
        "F1":       float(f1),
        "threshold":float(thr),
        "load_time_s":    load_time,
        "merge_time_s":   merge_time,
        "predict_time_s": predict_time,
        "n_samples"        : n_samples,
        "load_ms"          : load_time   * 1e3,
        "merge_ms"         : merge_time  * 1e3,
        "predict_ms"       : predict_time* 1e3,
        "total_ms"         : (load_time + merge_time + predict_time) * 1e3,
        "throughput_rows_s": n_samples / (load_time + merge_time + predict_time)
    }

    # ------------------------------------------------------------------ #
    # 5) BAGGING ensemble  (unchanged from your earlier code)            #
    # ------------------------------------------------------------------ #
    pred_start = time.perf_counter()
    bagging_pred = merged[featcols].mean(axis=1).to_numpy("float32")
    predict_time = time.perf_counter() - pred_start
    bag_auc  = roc_auc_score(y_true, bagging_pred)
    bag_ap   = average_precision_score(y_true, bagging_pred)
    bag_thr  = _best_threshold(y_true, bagging_pred)
    bag_p, bag_r, bag_f1, _ = precision_recall_fscore_support(
        y_true, (bagging_pred >= bag_thr).astype(int),
        average="binary", zero_division=0)

    bag_metrics = {
        "AUC":       float(bag_auc),
        "AP":        float(bag_ap),
        "precision": float(bag_p),
        "recall":    float(bag_r),
        "F1":        float(bag_f1),
        "threshold": float(bag_thr),
        "n_samples"        : n_samples,
        "load_ms"          : load_time   * 1e3,
        "merge_ms"         : merge_time  * 1e3,
        "predict_ms"       : predict_time* 1e3,
        "total_ms"         : (load_time + merge_time + predict_time) * 1e3,
        "throughput_rows_s": n_samples / (load_time + merge_time + predict_time)
    }
    #metrics.update(bag_metrics)                                # ← keep both sets together

    # ------------------------------------------------------------------ #
    # 6) write parquet & metrics                                         #
    # ------------------------------------------------------------------ #
    write_t0 = time.perf_counter()                              # ← added
    out_df = pd.DataFrame({
        "row_id":       np.arange(len(y_true), dtype="int64"),
        "ImpressionID": impr_ids,
        "y_true":       y_true,
        "y_pred":       y_pred
    })
    Path(store_parquet_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(store_parquet_path, engine="fastparquet",
                      compression="snappy", index=False)

    # extra k-metrics
    metrics.update(
        evaluate_parquet_scores(store_parquet_path, k_values=[5,10,20,50,100])
    )
    with open(store_metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    write_time = time.perf_counter() - write_t0                 # ← added
    metrics["write_time_s"]  = write_time                       # ← added
    metrics["total_runtime_s"] = time.perf_counter() - overall_start  # ← added

    if verbose:
        print(json.dumps(metrics, indent=2))

    # write bagging
    out_path_bag = Path(str(store_parquet_path).replace(".parquet", "_bagging.parquet"))
    out_path_bag_metrics = Path(str(store_metrics_path).replace(".json", "_bagging.json"))


    out_df = pd.DataFrame({
        "row_id":       np.arange(len(y_true), dtype="int64"),
        "ImpressionID": impr_ids,
        "y_true":       y_true,
        "y_pred":       bagging_pred,
    })
    Path(out_path_bag).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path_bag, engine="fastparquet",
                      compression="snappy", index=False)

    # extra k-metrics
    bag_metrics.update(
        evaluate_parquet_scores(out_path_bag, k_values=[5,10,20,50,100])
    )
    print(f"bag_metrics:{bag_metrics}")
    with open(out_path_bag_metrics, "w") as fh:
        json.dump(bag_metrics, fh, indent=2)


    if verbose:
        print(json.dumps(bag_metrics, indent=2))

    """
    if verbose:
        print(json.dumps(bag_metrics, indent=2))
    with open(out_path_bag_metrics, "w") as file:
        json.dump(bag_metrics, file, indent=4)

    if store_parquet_path is not None:
        out_path = Path(out_path_bag)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cols_to_write = ["row_id", "y_true", "y_pred", "ImpressionID"]
        merged[cols_to_write].to_parquet(
            out_path,
            engine="fastparquet",
            compression="snappy",
            index=False
        )
        if verbose:
            print(f"wrote meta predictions to {out_path}")
        bag_metrics_k = evaluate_parquet_scores(out_path, k_values=[5,10,20,50,100])
        bag_metrics = {**bag_metrics, **bag_metrics_k}
        print(json.dumps(bag_metrics, indent=2))
        with open(out_path_bag_metrics, "w") as file:
            json.dump(bag_metrics, file, indent=4)
    """


    return metrics


def evaluate_meta_from_parquet2(meta_path: str,
                                parquet_dir: str = "base_preds",
                                pattern: str = "*.parquet",
                                base_model_type = "all",
                                store_parquet_path="preds.parquet",
                                store_metrics_path="metrics.json",
                                verbose: bool = True,
                                booster='default',
                                batches=True):
    print(f"loading meta model from {meta_path}")
    meta = joblib.load(meta_path)
    if 'XGBClassifier' in meta_path:
        print(f"meta model .feature_importances_:{meta.feature_importances_}")
        booster = meta.get_booster()
        scores = booster.get_score(importance_type='weight')  
        print(f"meta model booster score weights:{scores}")
    if not 'XGBClassifier' in meta_path or booster == 'gblinear':
        print(f"meta model weights:{meta.coef_}")

    files = sorted(Path(parquet_dir).glob(pattern))
    if base_model_type == "cluster":
        #keep = re.compile(r"^(?:\d+|global)_")
        keep = re.compile(r"^(?:\d+)_")
        files = [fp for fp in files if keep.match(fp.stem)]
    if base_model_type == "category":
        remove = re.compile(r"^(?:\d+)_")
        files = [fp for fp in files if not remove.match(fp.stem)]
    if not files:
        raise FileNotFoundError(f"No parquet files matching {pattern}")

    print(f"base files:{files}")
    """
    merged   = None
    featcols = []

    for fp in files:
        #print(f"reading file:{fp}")
        #df  = pd.read_parquet(fp)
        cols = ["row_id", "y_true", "y_pred", "ImpressionID"]
        print(f"reading cols: {cols} from file:{fp}")
        #use_cols = cols if merged is None else ["row_id", "y_pred", "ImpressionID"]
        df = pd.read_parquet(fp, columns=cols)
        print(f"df.columns:{df.columns}")
        col = f"pred_{fp.stem}"
        df  = df.rename(columns={"y_pred": col})
        featcols.append(col)

        key_cols = ["row_id", "y_true"]
        if "ImpressionID" in df.columns:
            key_cols.append("ImpressionID")
        #merged = df[key_cols + [col]] if merged is None \
        #           else merged.merge(df[key_cols + [col]], on=key_cols, how="inner")
        if merged is None:
            merged = df[key_cols + [col]].copy()
        else:
            merged = merged.merge(
                df[['row_id', col]],
                on='row_id',
                how='inner'
            )

    if "row_id" not in merged.columns:
        merged.insert(0, "row_id", np.arange(len(merged), dtype="int64"))
    print(f"featcols:{featcols}")
    X      = merged[featcols].to_numpy(dtype="float32")
    y_true = merged["y_true"].to_numpy(dtype="int8")

    if batches:
        y_pred = gpu_predict_in_batches(meta, X, batch=200000)
    else:
        y_pred = meta.predict_proba(X)[:, 1]

    merged["y_pred"] = y_pred.astype("float32")
    """

    def load_and_prepare(fp, first=False):
        # on the very first file read impression too
        cols = ["row_id", "y_true", "y_pred", "ImpressionID"] if first \
            else ["row_id", "y_pred"]
        df = pd.read_parquet(fp, columns=cols)
        # rename the prediction column
        df = df.rename(columns={"y_pred": f"pred_{fp.stem}"})
        # set row_id as index so joins stay fast
        df = df.set_index("row_id")
        return df

    dfs = [ load_and_prepare(fp, first=(i==0))
            for i, fp in enumerate(files) ]

    # 2) join them all on row_id
    merged = reduce(lambda left, right: left.join(right, how="inner"), dfs)

    # 3) pull out arrays and column list
    y_true       = merged.pop("y_true").to_numpy(dtype="int8")
    impr_ids     = merged.pop("ImpressionID").to_numpy(dtype="int64")
    featcols     = merged.columns.tolist()
    X            = merged.to_numpy(dtype="float32")



    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    thr = _best_threshold(y_true, y_pred)
    print(f"_best_threshold:{thr}")
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, (y_pred >= thr).astype(int), average="binary", zero_division=0)

    metrics = {
        "samples":   int(len(y_true)),
        "AUC":       float(auc),
        "AP":        float(ap),
        "precision": float(p),
        "recall":    float(r),
        "F1":        float(f1)
    }

    if verbose:
        print(json.dumps(metrics, indent=2))
    with open(store_metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    if store_parquet_path is not None:
        out_path = Path(store_parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        """
        cols_to_write = ["row_id", "y_true", "y_pred", "ImpressionID"]
        merged[cols_to_write].to_parquet(
            out_path,
            engine="fastparquet",
            compression="snappy",
            index=False
        )
        """

        out = pd.DataFrame({
        "row_id":      np.arange(len(y_true), dtype="int64"),
        "ImpressionID": impr_ids,
        "y_true":      y_true,
        "y_pred":      y_pred  # from your meta.predict step
        })
        out.to_parquet(
            out_path,
            engine="fastparquet",
            compression="snappy",
            index=False
        )


        if verbose:
            print(f"wrote meta predictions to {out_path}")
        metrics_k = evaluate_parquet_scores(out_path, k_values=[5,10,20,50,100])
        metrics = {**metrics, **metrics_k}
        print(json.dumps(metrics, indent=2))
        with open(store_metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)



    bagging_pred = merged[featcols].mean(axis=1).to_numpy(dtype="float32")

    bag_auc = roc_auc_score(y_true, bagging_pred)
    bag_ap  = average_precision_score(y_true, bagging_pred)
    bag_thr = _best_threshold(y_true, bagging_pred)
    bag_p, bag_r, bag_f1, _ = precision_recall_fscore_support(
                y_true, (bagging_pred >= bag_thr).astype(int),
                average="binary", zero_division=0)

    bag_metrics = {
        "samples": int(len(y_true)),
        "bagging_AUC":       float(bag_auc),
        "bagging_AP":        float(bag_ap),
        "bagging_precision": float(bag_p),
        "bagging_recall":    float(bag_r),
        "bagging_F1":        float(bag_f1)
    }

    out_path_bag = Path(str(store_parquet_path).replace(".parquet", "_bagging.parquet"))
    out_path_bag_metrics = Path(str(store_metrics_path).replace(".json", "_bagging.json"))

    if verbose:
        print(json.dumps(bag_metrics, indent=2))
    with open(out_path_bag_metrics, "w") as file:
        json.dump(bag_metrics, file, indent=4)

    if store_parquet_path is not None:
        out_path = Path(out_path_bag)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cols_to_write = ["row_id", "y_true", "y_pred", "ImpressionID"]
        merged[cols_to_write].to_parquet(
            out_path,
            engine="fastparquet",
            compression="snappy",
            index=False
        )
        if verbose:
            print(f"wrote meta predictions to {out_path}")
        bag_metrics_k = evaluate_parquet_scores(out_path, k_values=[5,10,20,50,100])
        bag_metrics = {**bag_metrics, **bag_metrics_k}
        print(json.dumps(bag_metrics, indent=2))
        with open(out_path_bag_metrics, "w") as file:
            json.dump(bag_metrics, file, indent=4)
    return metrics

from collections import defaultdict
class RankingMetricsTracker:
    # Incrementally aggregates P@k, R@k, AP@k and nDCG@k over (y_true, y_score) pairs.

    def __init__(self, k_values=(5, 10, 20)):
        self.k_values  = sorted(k_values)
        self.hits      = defaultdict(int)
        self.recalls   = defaultdict(int)
        self.num_users = 0
        self.ap_sum    = defaultdict(float)
        self.ndcg_sum  = defaultdict(float)

    @staticmethod
    def _average_precision_at_k(rels, k):
        rels_k = rels[:k]
        if rels_k.sum() == 0:
            return 0.0
        precisions = np.cumsum(rels_k) / (np.arange(1, k + 1))
        return (precisions * rels_k).sum() / rels_k.sum()

    def update(self, y_true, y_score):
        if y_true.size < 2: # Need at least 2 rows for ordering
            return
        y_true  = np.asarray(y_true,  dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        order   = np.argsort(-y_score)
        rels    = y_true[order]

        num_rel = int(rels.sum())
        if num_rel == 0:
            return

        self.num_users += 1

        for k in self.k_values:
            k_eff = min(k, len(rels))
            hits_k = int(rels[:k_eff].sum())

            self.hits[k]    += hits_k
            self.recalls[k] += num_rel
            self.ap_sum[k]  += self._average_precision_at_k(rels, k_eff)

            self.ndcg_sum[k] += ndcg_score(
                y_true.reshape(1, -1),
                y_score.reshape(1, -1),
                k=k_eff
            )

    def result(self):
        metrics = {}
        for k in self.k_values:
            if self.num_users == 0:
                continue
            precision = self.hits[k] / (self.num_users * k)
            recall    = self.hits[k] / self.recalls[k]
            ap        = self.ap_sum[k]  / self.num_users
            ndcg      = self.ndcg_sum[k] / self.num_users
            metrics[k] = {"precision": precision,
                          "recall":    recall,
                          "ap":        ap,
                          "ndcg":      ndcg}
        return metrics

def evaluate_parquet_scores(parquet_path, k_values = (5, 10, 20, 50), key="ImpressionID"):
    # Computes ranking metrics from a saved .parquet prediction file.
    # The file must row_id, y_true and y_pred columns

    df = pd.read_parquet(parquet_path)
    print(f"parquet file: {parquet_path}, columns: {df.columns.tolist()}")
    # If grouped by user

    group_key = key if key in df.columns else None
    if group_key != key:
        log_print(f"key:{key} is not found in columns!!!!!!!!!!!!!!!!!!!\nkey:{key} is not found in columns!!!!!!!!!!!!!!!!!!!\nkey:{key} is not found in columns!!!!!!!!!!!!!!!!!!!\n")
    tracker = RankingMetricsTracker(k_values=k_values)

    if group_key:
        for _, g in df.groupby(group_key, sort=False):
            tracker.update(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
    else:
        tracker.update(df["y_true"].to_numpy(), df["y_pred"].to_numpy())

    return tracker.result()

def train_test_split_time(behaviors_df, cutoff_str="2019-11-20"):
    # Ensure Time column is datetime64[ns] and timezone-naive
    if behaviors_df["Time"].dtype != "datetime64[ns]":
        behaviors_df["Time"] = pd.to_datetime(behaviors_df["Time"], errors="coerce")
    if behaviors_df["Time"].dt.tz is not None:
        behaviors_df["Time"] = behaviors_df["Time"].dt.tz_localize(None)

    # Also ensure cutoff_dt is timezone-naive
    cutoff_dt = pd.to_datetime(cutoff_str)
    if cutoff_dt.tzinfo is not None:
        cutoff_dt = cutoff_dt.tz_localize(None)

    train_data = behaviors_df[behaviors_df["Time"] <= cutoff_dt].copy()
    test_data = behaviors_df[behaviors_df["Time"] > cutoff_dt].copy()
    return train_data, test_data

def filter_csv_file(input_csv, keep_donor=False, output_csv=None):
    # Reads input_csv and filters out rows where 'num_future_clicks' is 0 as well as row with or without donor.
    #fixed_csv = fix_header_only(input_csv)
    df = pd.read_csv(input_csv)
    print(f"read:{input_csv}")
    #print("Unique donor_user_id values:", df["donor_user_id"].unique())
    #print("True NaNs:", df["donor_user_id"].isnull().sum())
    #print("Empty strings:", (df["donor_user_id"] == "").sum())
    #df["donor_user_id"].replace("", np.nan, inplace=True)
    df = df[df["num_future_clicks"] != 0]
    if keep_donor:
        df = df[df["donor_user_id"].notnull()]
    else:
        df = df[~df["donor_user_id"].notnull() | ~df["num_from_original"].notnull()]
    df = df.drop_duplicates(subset="user_id", keep="last")

    if output_csv is None:
        dir_name, base_name = os.path.split(input_csv)
        if keep_donor:
            base_name = f"{base_name.split('.csv')[0]}_keep_donor.csv"
        else:
            base_name = f"{base_name.split('.csv')[0]}_no_donor.csv"
        output_csv = os.path.join(dir_name, f"filtered_{base_name}")

    df.to_csv(output_csv, index=False)
    print(f"Filtered CSV saved as: {output_csv}")
    return output_csv

def combine_csv_files(input_pattern, combined_output_file=None):
    #Combines all CSV files that match input_pattern into a single CSV file.

    file_list = glob.glob(input_pattern)
    if not file_list:
        raise ValueError("No files found matching the given pattern.")
    
    df_list = [pd.read_csv(f) for f in file_list]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    dates = []
    for f in file_list:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 7:
            date_str = parts[5]
            dates.append(date_str)
    unique_dates = sorted(set(dates))
    if combined_output_file is None:
        # Assume all files are for the same model_key and cluster.
        basename = os.path.basename(file_list[0])
        parts = basename.split('_')
        model_key = parts[0]
        cluster_num = os.path.splitext(parts[-1])[0]
        if len(unique_dates) == 1:
            date_part = unique_dates[0]
        else:
            date_part = f"{unique_dates[0]}-{unique_dates[-1]}"
        combined_output_file = f"{model_key}_user_level_partial_results_{date_part}_{cluster_num}.csv"
    
    combined_df.to_csv(combined_output_file, index=False)
    print(f"Combined CSV saved as: {combined_output_file}")
    return combined_output_file

def compute_experiment_summary(csv_files, k_values=[5, 10, 20, 50, 100], auc=False):
    # If a glob pattern is provided, collect all matching files.
    if isinstance(csv_files, str):
        files = glob.glob(csv_files)
    else:
        files = csv_files
    df_list = [pd.read_csv(f) for f in files]


    combined_df = pd.concat(df_list, ignore_index=True)
    
    #for f in combined_df:
    #    f["method"] = f["file"].split("_")[1]
    # Optionally, filter out rows where num_future_clicks == 0.
    combined_df = combined_df[combined_df["num_future_clicks"] != 0]
    
    # Initialize a dictionary to hold the summary results.
    summary_dict = {}
    summary_dict["total_users"] = combined_df.shape[0]
    print(combined_df["num_future_clicks"])
    print(combined_df["num_shown_articles"])
    print(combined_df["num_candidates_full"])
    summary_dict["avg_num_future_clicks"] = combined_df["num_future_clicks"].mean()
    summary_dict["avg_num_shown_articles"] = combined_df["num_shown_articles"].mean() # this column might have real num_future_clicks value!

    print(f"avg_num_shown_articles column might actually have real avg_num_future_clicks value!!!!!")
    print(f"avg_num_future_clicks might actually have real avg_num_candidates_full value!!!!!")
    print(f"avg_num_shown_articles column might actually have real avg_num_future_clicks value!!!!!")
    print(f"avg_num_future_clicks might actually have real avg_num_candidates_full value!!!!!")
    print(f"avg_num_shown_articles column might actually have real avg_num_future_clicks value!!!!!")
    print(f"avg_num_future_clicks might actually have real avg_num_candidates_full value!!!!!")
    print(f"avg_num_shown_articles column might actually have real avg_num_future_clicks value!!!!!")
    print(f"avg_num_future_clicks might actually have real avg_num_candidates_full value!!!!!")
    print(f"avg_num_shown_articles column might actually have real avg_num_future_clicks value!!!!!")
    print(f"avg_num_future_clicks might actually have real avg_num_candidates_full value!!!!!")
    print(f"avg_num_shown_articles column might actually have real avg_num_future_clicks value!!!!!")
    print(f"avg_num_future_clicks might actually have real avg_num_candidates_full value!!!!!")

    summary_dict["avg_num_candidates_full"] = combined_df["num_candidates_full"].mean()
    summary_dict["avg_num_history_articles"] = combined_df["num_history_articles"].mean()
    if auc:
        summary_dict["avg_auc_full"]     = combined_df["auc_full"].mean(skipna=True)
        summary_dict["avg_auc_behavior"] = combined_df["auc_behavior"].mean(skipna=True)
    # For each k value, calculate the average metrics for 'full' candidate pool.
    for k in k_values:
        prec_col = f"precision_full_{k}"
        rec_col = f"recall_full_{k}"
        ap_col = f"ap_full_{k}"
        ndcg_col = f"ndcg_full_{k}"
        num_rec_col = f"num_recommendations_full_{k}"
        
        summary_dict[f"avg_precision_full_{k}"] = combined_df[prec_col].mean()
        summary_dict[f"avg_recall_full_{k}"] = combined_df[rec_col].mean()
        summary_dict[f"avg_ap_full_{k}"] = combined_df[ap_col].mean()
        summary_dict[f"avg_ndcg_full_{k}"] = combined_df[ndcg_col].mean()
        summary_dict[f"avg_num_recommendations_full_{k}"] = combined_df[num_rec_col].mean()
        
    # Optionally, compute similar statistics for the 'behavior' candidate pool.
    for k in k_values:
        prec_col = f"precision_behavior_{k}"
        rec_col = f"recall_behavior_{k}"
        ap_col = f"ap_behavior_{k}"
        ndcg_col = f"ndcg_behavior_{k}"
        num_rec_col = f"num_recommendations_behavior_{k}"
        
        summary_dict[f"avg_precision_behavior_{k}"] = combined_df[prec_col].mean()
        summary_dict[f"avg_recall_behavior_{k}"] = combined_df[rec_col].mean()
        summary_dict[f"avg_ap_behavior_{k}"] = combined_df[ap_col].mean()
        summary_dict[f"avg_ndcg_behavior_{k}"] = combined_df[ndcg_col].mean()
        summary_dict[f"avg_num_recommendations_behavior_{k}"] = combined_df[num_rec_col].mean()
    
    # Create a summary DataFrame from the dictionary.
    summary_df = pd.DataFrame.from_dict(summary_dict, orient="index", columns=["Value"])
    #print(summary_df["file"])
    return summary_df, summary_dict

def build_and_save_title_tensors(
    news_df,
    tokenizer,
    max_title_length: int,
    out_path: str = "title_tensors.pkl"
):
    news_ids = news_df["NewsID"].tolist()
    titles   = news_df["Title"].fillna("").tolist()

    sequences = tokenizer.texts_to_sequences(titles)

    padded = pad_sequences(
        sequences,
        maxlen=max_title_length,
        padding="post",
        truncating="post",
        value=0
    )
    id_to_index = {nid: idx for idx, nid in enumerate(news_ids)}
    with open(out_path, "wb") as f:
        pickle.dump({
            "news_ids": news_ids,
            "padded": padded,
            "id_to_index": id_to_index
        }, f)

    print(f"Saved {len(news_ids)} title tensors to {out_path}")

def get_title_tensors(dataset, dataset_size, news_df, tokenizer, max_title_length=30, make_new=False):
    tensors_file = f"{dataset}_{dataset_size}_title_tensors.pkl"
    log_print(f"tensors_file:{tensors_file}")
    if make_new or not os.path.exists(tensors_file):
        build_and_save_title_tensors(
            news_df=news_df,
            tokenizer=tokenizer,
            max_title_length=max_title_length,
            out_path=tensors_file
        )

    with open(tensors_file, "rb") as f:
        data = pickle.load(f)
    padded = data["padded"]
    title_tensor = tf.convert_to_tensor(padded, dtype=tf.int32)
    id_to_index = data["id_to_index"]
    return title_tensor, id_to_index

def main(dataset='train', process_dfs=False, process_behaviors=False,
        data_dir_train='dataset/train/', data_dir_valid='dataset/valid/',
        zip_file_train="MINDlarge_train.zip", zip_file_valid="MINDlarge_dev.zip",
        user_category_profiles_path='', user_cluster_df_path='', cluster_id=None, meta_train=False,resume=True,
        model_type="cluster", dataset_size='large', load_best_model=False, load_best_models=[], eval_scope='cluster', model_size='large', epochs=1,
        adaptivity_test=False, donor_strategy='random', shuffle=False, drift_fraction=0.5, use_full_set=True, eval_separate=False, use_full_eval_separate_set=False,
        skip_already_evaluated=False, batch_size=256, retrain_models=[], eval_dataset_size='large',
        ext_data_dir_train='dataset/train/', ext_data_dir_valid='dataset/valid/',
        ext_zip_file_train="MINDlarge_train.zip", ext_zip_file_valid="MINDlarge_dev.zip",n_estimators=300, test_size=0.2, end_after_process=False,
        use_model_base=False, valid_dataset_size='large', use_avg_profile=False, meta_train_dataset='valid', pad_zeros=True, use_full_avg_profile=False, process_valid_sets=False,
        make_title_tensor=False,eval_frac=1.0, big_tokenizer=False, pivots=[45], ks=[5], reverse=False, tune_threshold=False, regenerate_metrics=False, vers_suffix="_v4", model_arc_type="fastformer",
        timed=False, use_cpu=False, clear_store=False):
    # Main function to run tests on a given dataset type ('train' or 'valid').
    # It uses the midpoint time as cutoff and then runs evaluations.
    # Choose dataset directory based on parameter.
    if use_cpu: # Hide gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("wait 4 hours")
    #time.sleep(14400)
    print("wait ended")
    if dataset.lower() == 'train':
        data_dir = data_dir_train
        ext_data_dir = ext_data_dir_train
    elif dataset.lower() == 'valid':
        data_dir = data_dir_valid
        ext_data_dir = ext_data_dir_valid
    else:
        raise ValueError("dataset must be either 'train' or 'valid'")
    
    # Unzip and load dataset
    unzip_datasets(data_dir_train, data_dir_valid, zip_file_train, zip_file_valid)
    news_df, behaviors_df = init_dataset(data_dir)
    unzip_datasets(ext_data_dir_train, ext_data_dir_valid, ext_zip_file_train, ext_zip_file_valid)
    ext_news_df, ext_behaviors_df = init_dataset(ext_data_dir)
    print(f"behaviors_df:{behaviors_df}")
    tokenizer, vocab_size = prepare_tokenizer(news_df)
    max_history_length = 50
    max_title_length = 30

    midpoint_time = get_midpoint_time(behaviors_df)
    ext_midpoint_time = get_midpoint_time(ext_behaviors_df)
    # Format the time to ISO format with a trailing 'Z'
    #cutoff_time_str = midpoint_time.isoformat().replace('+00:00', 'Z')
    cutoff_time_str = ext_midpoint_time.isoformat().replace('+00:00', 'Z')
    print("Using cutoff time:", cutoff_time_str)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(news_df["CombinedText"])
    
    full_data = ext_behaviors_df
    # train_test_split function gets train/test splits from behaviors_df.
    if use_full_set:
        if dataset.lower() == 'train':
            train_data = behaviors_df
            test_data = pd.DataFrame([{}])
        elif dataset.lower() == 'valid':
            test_data = behaviors_df
            train_data = pd.DataFrame([{}])
    else:
        train_data, test_data = train_test_split_time(behaviors_df, cutoff_time_str)

    if model_type == "cluster":
        models_dict, news_df, behaviors_df, tokenizer = get_models(process_dfs, process_behaviors, data_dir_train, data_dir_valid, zip_file_train, zip_file_valid, evaluate=eval_separate, dataset=dataset,
        skip_already_evaluated=skip_already_evaluated, model_size=model_size, batch_size=batch_size, retrain_models=retrain_models, dataset_size=dataset_size,
        cutoff_time_str=cutoff_time_str, eval_full=use_full_eval_separate_set, test_size=test_size, use_model_base=use_model_base, process_valid_sets=process_valid_sets, make_title_tensor=make_title_tensor,
        eval_frac=eval_frac, big_tokenizer=big_tokenizer, pivots=pivots, ks=ks, reverse=reverse, tune_threshold=tune_threshold, regenerate_metrics=regenerate_metrics, vers_suffix=vers_suffix, model_arc_type=model_arc_type,
        timed=timed, use_cpu=use_cpu, clear_store=clear_store)
    if model_type == "category" or model_type == "all" or model_type == "global":
        models_dict, news_df, behaviors_df, tokenizer = get_models(process_dfs, process_behaviors, data_dir_train, data_dir_valid, zip_file_train, zip_file_valid, model_type=model_type, dataset_size=dataset_size,
        model_size=model_size, load_best_model=load_best_model, load_best_models=load_best_models, epochs=epochs, evaluate=eval_separate, dataset=dataset,
        skip_already_evaluated=skip_already_evaluated, batch_size=batch_size, retrain_models=retrain_models, cutoff_time_str=cutoff_time_str, eval_full=use_full_eval_separate_set, test_size=test_size,
        use_model_base=use_model_base, process_valid_sets=process_valid_sets, make_title_tensor=make_title_tensor, eval_frac=eval_frac, big_tokenizer=big_tokenizer, pivots=pivots, ks=ks, reverse=reverse,
        tune_threshold=tune_threshold, regenerate_metrics=regenerate_metrics, vers_suffix=vers_suffix, model_arc_type=model_arc_type, timed=timed, use_cpu=use_cpu, clear_store=clear_store)
    #if end_after_process:
    log_print(f"ending")
    return
    print(f"get_models give badd behaviors_df!!")
    print(f"behaviors_df:{behaviors_df}")
    news_df, behaviors_df = init_dataset(data_dir)
    print(f"behaviors_df:{behaviors_df}")

    if user_cluster_df_path == '':
        user_cluster_df_path = hf_hub_download(
            repo_id="Teemu5/news",
            filename="user_cluster_df.pkl",
            local_dir="models"
        )
    else:
        make_user_cluster_df(user_category_profiles_path, user_cluster_df_path)
    user_cluster_df = pd.read_pickle(user_cluster_df_path)
    cluster_mapping = {}
    # Check what (cluster) to train on
    for cluster in user_cluster_df['Cluster'].unique():
        cluster_mapping[cluster] = user_cluster_df[user_cluster_df['Cluster'] == cluster].index.tolist()
    clusters_to_run = [999]
    if cluster_id is not None:
        clusters_to_run = [int(x.strip()) for x in str(cluster_id).split(",")]
        cluster_mapping = {cl: users for cl, users in cluster_mapping.items() if cl in clusters_to_run}
        print("Processing only clusters:", list(cluster_mapping.keys()))
    all_users = list(full_data["UserID"].unique())
    if eval_scope == "global":
        cluster_mapping = {"ALL_USERS": all_users}

    meta_name = f"XGBClassifier_hist_{n_estimators}"
    meta_model_base = f"meta_model_{meta_name}_{model_type}_{model_size}_{meta_train_dataset}_{dataset_size}"
    meta_model_base_pattern = f"meta_model_{meta_name}_*_{model_size}_{meta_train_dataset}_{dataset_size}"
    if test_size >= 0.0:
        meta_model_base = f"meta_model_{meta_name}_{model_type}_{model_size}_train_{dataset_size}_test_size_{test_size}"
        meta_model_base_pattern = f"meta_model_{meta_name}_*_{model_size}_train_{dataset_size}_test_size_{test_size}"
    if eval_dataset_size != dataset_size:
        log_print(f"!!! eval_dataset_size != dataset_size !!! eval_dataset_size:{eval_dataset_size} and dataset_size:{dataset_size}")
        log_print(f"!!! eval_dataset_size != dataset_size !!! eval_dataset_size:{eval_dataset_size} and dataset_size:{dataset_size}")
        log_print(f"!!! eval_dataset_size != dataset_size !!! eval_dataset_size:{eval_dataset_size} and dataset_size:{dataset_size}")
        dataset_size = eval_dataset_size
    # Results are written into results_partial_csv during test run
    results_partial_csv = f"{model_size}_{model_type}_{dataset}_{dataset_size}_user_level_partial_results_{date_str}_{list(cluster_mapping.keys())[0]}.csv"
    if load_best_model:
        results_partial_csv = f"best_model_{results_partial_csv}"
    if adaptivity_test:
        results_partial_csv = f"{donor_strategy}_{drift_fraction}_{results_partial_csv}"
    ext_model_type = f"{model_type}"
    if load_best_model:
        ext_model_type = f"{model_type}_best_model"

    filename = f"global_average_profile_{dataset_size}.pkl"
    if os.path.exists(filename) and use_avg_profile:
        with open(filename, "rb") as f:
            avg_profile = pickle.load(f)
        print(f"Loaded global average profile from file:{avg_profile}")
        logging.info(f"Loaded global average profile from file:{avg_profile}")
    else:
        avg_profile = None

    cutoff = pd.to_datetime(cutoff_time_str)
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    profiles_file = f"{dataset}_{dataset_size}_user_profiles.pkl"
    if not os.path.exists(profiles_file):
        all_users = [u for cl in cluster_mapping.values() for u in cl]
        build_user_profiles(
            user_ids=all_users,
            behaviors_df=test_data,
            news_df=news_df,
            tokenizer=tokenizer,
            avg_profile=avg_profile,
            cutoff_time=cutoff_time_str,
            max_history_length=max_history_length,
            max_title_length=max_title_length,
            out_path=profiles_file,
            cutoff=cutoff
        )
    with open(profiles_file, "rb") as f:
        user_profiles = pickle.load(f)
    
    title_tensor, id_to_index = get_title_tensors(dataset, dataset_size, news_df, tokenizer, make_new=make_title_tensor)
    #meta_model_XGBClassifier_hist_cluster_small_train_small
    meta_model_path = f"meta/{meta_model_base}.joblib"
    meta_model_pattern = f"meta/{meta_model_base_pattern}.joblib"


    #all_users = [u for users in cluster_mapping.values() for u in users if u != user_id]
    viable_donors_file = f"viable_donors_full_large_{cutoff_time_str}.pkl"
    if 'small' in ext_data_dir:
        viable_donors_file = f"viable_donors_full_small_{cutoff_time_str}.pkl"
    
    def get_viable_donors(all_users, behaviors_df, cutoff,
                    cache="viable_donors.pkl", every=1000):
        p = Path(cache)
        if p.exists():
            with p.open("rb") as fh:
                return pickle.load(fh)

        donors, total = [], len(all_users)
        for i, u in enumerate(tqdm(all_users, desc="Scanning users"), 1):
            if get_user_history_ids(u, behaviors_df, cutoff):
                donors.append(u)
            if i % every == 0:
                print(f"{i}/{total} checked – {len(donors)} donors")

        with p.open("wb") as fh:
            pickle.dump(donors, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return donors
    viable_donors = get_viable_donors(all_users, behaviors_df, cutoff, cache=viable_donors_file, every=1000)
    """
    viable_donors_file = pathlib.Path(viable_donors_file)
    if viable_donors_file.exists():
        with viable_donors_file.open("rb") as f:
            viable_donors = pickle.load(f)
    else:
        viable_donors = [
            u for u in all_users
            if get_user_history_ids(u, behaviors_df, cutoff_time_str)
        ]
        with viable_donors_file.open("wb") as f:
            pickle.dump(viable_donors, f, protocol=pickle.HIGHEST_PROTOCOL)
    """
    print(f"{len(viable_donors)} viable donors")

    results_user_level = run_experiments_user_level(
        cluster_mapping,
        viable_donors,
        train_data,
        test_data,
        news_df,
        behaviors_df,
        models_dict,
        tokenizer,
        tfidf_vectorizer,
        cutoff_time_str,cutoff,
        k_values=[5, 10, 20, 50, 100],
        partial_csv=results_partial_csv,
        shuffle_clusters=False,
        resume=resume,
        model_type=ext_model_type,
        dataset_size=dataset_size,
        model_size=model_size,
        donor_strategy=donor_strategy,
        adaptivity_test=adaptivity_test,
        shuffle_user=shuffle,
        drift_fraction=drift_fraction,
        avg_profile=avg_profile,
        user_profiles=user_profiles,
        id_to_index=id_to_index,
        title_tensor=title_tensor,
        meta_model_path=meta_model_path,
        meta_model_pattern=meta_model_pattern,
        pad_zeros=pad_zeros,
        use_full_avg_profile=use_full_avg_profile
    )
    
    print("Evaluation complete. Intermediate results were written during testing.")
    return results_user_level, cluster_results_df
