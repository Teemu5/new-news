
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
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, register_keras_serializable, custom_object_scope
from keras.models import Model, model_from_json

from huggingface_hub import hf_hub_download

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
        s if s else [tokenizer.oov_token or 1]  # avoid all-zero row
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

def get_candidate_pool_for_user(user_id, behaviors_df, news_df, cutoff_time_str, cutoff, user_history_ids):
    global CANDIDATE_POOL_CACHE
    if cutoff_time_str not in CANDIDATE_POOL_CACHE:
        CANDIDATE_POOL_CACHE[cutoff_time_str] = precompute_candidate_pool(behaviors_df, cutoff)
    candidate_pool = CANDIDATE_POOL_CACHE[cutoff_time_str]
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
    id_to_index=None,title_tensor=None
):
    """
    Generating a candidate pool for a single user.
    1. gather all articles that exist up to cutoff_time.
    2. Exclude articles in user_history_ids.
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
    drift_fraction=0,user_profiles=None,id_to_index=None,title_tensor=None
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

    user_history_tensor, used_ids, original_history_len = build_user_profile_tensor(
        user_id=user_id,
        behaviors_df=behaviors_df,
        news_df=news_df,
        cutoff_time_str=cutoff_time_str,
        cutoff=cutoff,
        tokenizer=tokenizer,
        avg_profile=avg_profile,
        pad_zeros=True,
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
                                    encoder_path = None,meta_model_pattern=None):
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
                    id_to_index=id_to_index, title_tensor=title_tensor
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
                        title_tensor=title_tensor
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



def prepare_train_df(
    data_dir,
    news_file,
    behaviors_file,
    user_category_profiles,
    num_clusters=3,
    fraction=1,
    max_title_length=30,
    max_history_length=50,
    downsampling=False,
    categorized_samples=False,
    news_df_pkl="models/news_df_processed.pkl", train_df_pkl="models/train_df_processed.pkl"
    ):
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    print("Loaded news data:")
    print(news_df.head())

    behaviors_path = os.path.join(data_dir, behaviors_file)
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    print("Loaded behaviors data:")
    print(behaviors_df.head())

    news_df['CleanTitle'] = news_df['Title'].apply(clean_text)
    news_df['CleanAbstract'] = news_df['Abstract'].apply(clean_text)
    news_df['CombinedText'] = news_df['CleanTitle'] + ' ' + news_df['CleanAbstract']
    news_df["CombinedText"] = news_df["CombinedText"].astype(str)
    news_df["CombinedText"] = news_df["CombinedText"].fillna("")

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    news_df['EncodedText'] = tokenizer.texts_to_sequences(news_df['CombinedText'])
    news_df['PaddedText'] = list(pad_sequences(news_df['EncodedText'], maxlen=max_title_length, padding='post', truncating='post'))
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

    # Parse news ids and labels behaviors data
    behaviors_df[['ImpressionNewsIDs', 'ImpressionLabels']] = behaviors_df['Impressions'].apply(
        lambda x: pd.Series(parse_impressions(x))
    )

    train_samples = []
    # Iterate over behaviors to create train samples
    for _, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0]):
        user_id = row['UserID']
        user_cluster = row['Cluster'] if 'Cluster' in row else None

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

        for candidate_news_id, label in zip(candidate_news_ids, labels):
            candidate_text = news_text_dict.get(candidate_news_id, [0]*max_title_length)
            sample = {
                'UserID': user_id,
                'HistoryTitles': history_texts,
                'CandidateTitleTokens': candidate_text,
                'Label': label
            }
            if categorized_samples:
                candidate_category_series = news_df[news_df['NewsID'] == candidate_news_id]['Category']
                candidate_category = candidate_category_series.iloc[0] if not candidate_category_series.empty else "Unknown"
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
    # Optionally perform random sampling
    print(f"Original size: {len(train_df)}")
    train_df_sampled = train_df.sample(frac=fraction, random_state=42)
    print(f"Sampled size: {len(train_df_sampled)}")

    # Optionally, set train_df to sampled
    train_df = train_df_sampled
    print("Columns in sampled train_df:")
    print(train_df.columns)
    print(f"Cluster:{train_df['Cluster']}")
    print("Splitting data into training and validation sets for each cluster...")
    clustered_data = {}
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]

        if cluster_data.empty:
            print(f"No data for Cluster {cluster}. Skipping...")
            continue  # Skip to the next cluster

        train_data, val_data = train_test_split(cluster_data, test_size=0.2, random_state=42, stratify=None)
        clustered_data[cluster] = {
            'train': train_data.reset_index(drop=True),
            'val': val_data.reset_index(drop=True)
        }
        print(f"Cluster {cluster}: {len(train_data)} training samples, {len(val_data)} validation samples.")
    print(f"Saved after processing: {news_df_pkl}, {train_df_pkl}")
    news_df.to_pickle(news_df_pkl)
    train_df.to_pickle(train_df_pkl)

    return clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters

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

def build_model(vocab_size, max_title_length=30, max_history_length=50, embedding_dim=256, nb_head=8, size_per_head=32, dropout_rate=0.2):
    # Define Inputs
    history_input = Input(shape=(max_history_length, max_title_length), dtype='int32', name='history_input')
    candidate_input = Input(shape=(max_title_length,), dtype='int32', name='candidate_input')

    # Instantiate NewsEncoder Layer
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

def train_cluster_models(clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters, batch_size=64, epochs=5, load_models=[], retrain=False, size='large'):
    models = {}
    model_dir = "models"    
    for cluster in range(num_clusters):
        m_name = f'fastformer_cluster_{cluster}_{size}_full_balanced_1_epoch'
        weights_file = f'{model_dir}/{m_name}.weights.h5'
        model_file = f'{model_dir}/{m_name}.keras'
        model_h5_file = f'{model_dir}/{m_name}.h5'
        model_hdf5_file = f'{model_dir}/{m_name}.hdf5'
        model_json_file = f'{model_dir}/{m_name}.json'
        #if cluster in load_models: # load_models should be list of number indicating which models to load and not train
        #    print(f"\nLoading model for Cluster {cluster} from {model_file}")
        #    local_model_path = hf_hub_download(
        #        repo_id=f"Teemu5/news",
        #        filename=model_file,
        #        local_dir=model_dir
        #    )
        model_path = model_h5_file
        if os.path.exists(model_path) and not retrain:
            print(f"Loading model: {model_path}")
            print(tf.__version__)
            print(keras.__version__)
            with custom_object_scope({'UserEncoder': UserEncoder, 'NewsEncoder': NewsEncoder}):
                model = tf.keras.models.load_model(model_path)#build_and_load_weights(weights_file)
                models[f"{cluster}_{size}"] = model
            #model.save(model_file)
            #print(f"Saved model for Cluster {cluster} into {model_file}.")
            continue
        
        model_file = f"{model_dir}/{model_file}"
        print(f"\nTraining model for Cluster {cluster} into {weights_file}")
        # Retrieve training and validation data
        train_data = clustered_data[cluster]['train']
        val_data = clustered_data[cluster]['val']

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
        model.save_weights(weights_file)
        print(f"Saved model weights for Cluster {cluster} into {weights_file}.")
        model.save(model_h5_file)
        print(f"Saved h5 model for Cluster {cluster} into {model_h5_file}.")
        model.save(model_file)
        print(f"Saved model for Cluster {cluster} into {model_file}.")

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


def resume_or_build_h5(model_dir, build_fn, steps_per_epoch):
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
    model = load(latest, custom_objects=None)
    return model, start_epoch

def train_global_model(
    train_df,
    tokenizer,
    vocab_size,
    max_history_length,
    max_title_length,
    dataset_size,
    batch_size=128,
    epochs=5,
    retrain=False,
    load_best_model=False,
    load_checkpoint_model=False
):
    model_file_prefix=f"fastformer_global_{dataset_size}_balanced_{epochs}_epochs"
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
        model = load(load_file)
        print(f"Loaded existing global model from {keras_file}")
        return model
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

    train_generator = DataGenerator(train_data, batch_size=batch_size, max_history_length=max_history_length, max_title_length=max_title_length)
    val_generator = DataGenerator(val_data, batch_size=batch_size, max_history_length=max_history_length, max_title_length=max_title_length)
    steps_per_epoch = len(train_generator)

    model, start_epoch = resume_or_build_h5(
        model_dir=model_dir,
        build_fn=lambda: build_model(
            vocab_size=vocab_size,
            max_title_length=max_title_length,
            max_history_length=max_history_length,
            embedding_dim=256,
            nb_head=8,
            size_per_head=32,
            dropout_rate=0.2
        ),
        steps_per_epoch=steps_per_epoch
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
    model.fit(
        train_generator,
        epochs=epochs,
        initial_epoch=start_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
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
def evaluate_with_generator(model,
                            eval_df,
                            batch_size      = 128,
                            max_history_len = 50,
                            max_title_len   = 30,
                            store_path      = None,
                            flush_every     = 0.10,
                            verbose         = 1,
                            store_metrics_path="metrics.json"):

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
        X, y_true = gen[b]
        if X is None:
            continue

        y_pred = model.predict(X, batch_size=batch_size, verbose=0).squeeze()

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        if store_path is not None:
            for lbl, pred in zip(y_true, y_pred):
                buf.append({'row_id': seen,
                            'y_true': float(lbl),
                            'y_pred': float(pred)})
                seen += 1

            if len(buf) >= flush_interval * batch_size:
                _flush(buf, store_path)
                written += len(buf);  buf.clear()

        if (b + 1) % flush_interval == 0:
            y_true_so_far = np.concatenate(y_true_all, dtype="float32")
            y_pred_so_far = np.concatenate(y_pred_all, dtype="float32")

            auc_so_far = roc_auc_score(y_true_so_far, y_pred_so_far)
            p, r, f, _ = precision_recall_fscore_support(
                            y_true_so_far,
                            (y_pred_so_far >= 0.5).astype(int),
                            average="binary",
                            zero_division=0)

            if verbose:
                print(
                    f"[{b+1:>4}/{total_batches}] "
                    f"AUC={auc_so_far:0.4f}  P={p:0.4f}  R={r:0.4f}  F1={f:0.4f}"
                )

    if buf and store_path is not None:
        _flush(buf, store_path)
        written += len(buf)

    if verbose and store_path is not None:
        print(f"Stored {written:,} prediction rows to {store_path}")

    y_true_all = np.concatenate(y_true_all, dtype="float32")
    y_pred_all = np.concatenate(y_pred_all, dtype="float32")

    metrics = {"samples": int(len(y_true_all)),
               "AUC": float(roc_auc_score(y_true_all, y_pred_all))}
    p, r, f, _ = precision_recall_fscore_support(
                    y_true_all, (y_pred_all >= 0.5).astype(int),
                    average="binary", zero_division=0)
    metrics.update({'precision': float(p),
                    'recall':    float(r),
                    'F1':        float(f)})

    if verbose:
        print(json.dumps(metrics, indent=2))
    with open(store_metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    if store_path is not None:
        metrics_k = evaluate_parquet_scores(store_path, k_values=[5,10,20,50,100])
        metrics = {**metrics, **metrics_k}

    if verbose:
        print("Evaluation summary")
        print(json.dumps(metrics, indent=2))
    with open(store_metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    return metrics

def _flush(buffer, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(buffer).to_parquet(path, engine="fastparquet",
                                    compression="snappy",
                                    index=False,
                                    append=os.path.exists(path))


def load(load_this, custom_objects=None, max_history_length=50, max_title_length=30):
    logging.info(f"Loading model: {load_this}")
    print(f"Loading model: {load_this}")
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

def train_category_models(category_train_dfs, vocab_size, max_history_length, max_title_length, batch_size=64, epochs=5, dataset_size='', train_only_new=True, train_fraction=1.0, load_best_model=True):
    #Train a model for each category in the category_train_dfs dict.
    model_dir = "models"
    category_models = {}
    for category, df in category_train_dfs.items():
        print(f"Training {category.upper()}")
        print(df)
        train_df, val_df = df["train"], df["val"]
        keras_model_save_path = f'{model_dir}/fastformer_{dataset_size}_category_{category}_{epochs}epochs.keras'
        model_save_path = f'{model_dir}/fastformer_{dataset_size}_category_{category}_{epochs}epochs.h5'
        best_model = f'{model_dir}/best_model_fastformer_{dataset_size}_category_{category}_{epochs}epochs.h5'
        keras_best_model = f'{model_dir}/fastformer_{dataset_size}_category_{category}_{epochs}epochs.keras'
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
        train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
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
def make_clustered_data(train_df, num_clusters, test_size=0.2, random_state=42):
    clustered_data = {}
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]

        if cluster_data.empty:
            print(f"No data for Cluster {cluster}. Skipping...")
            continue

        train_data, val_data = train_test_split(cluster_data, test_size=test_size, random_state=random_state, stratify=None)
        clustered_data[cluster] = {
            'train': train_data.reset_index(drop=True),
            'val': val_data.reset_index(drop=True)
        }
        print(f"Cluster {cluster}: {len(train_data)} training samples, {len(val_data)} validation samples.")
    return clustered_data

def make_category_data(train_df, test_size=0.2, random_state=42):
    # Returns a dict like {'Sports': {'train': …, 'val': …}, …}
    category_data = {}
    for cat, cat_df in train_df.groupby("CandidateCategory"):
        print(f"cat:{cat}, cat_df:{cat_df}")
        if cat_df.empty:
            continue
        tr, val = train_test_split(cat_df,
                                   test_size=test_size,
                                   random_state=random_state,
                                   stratify=None)
        category_data[cat] = {
            "train": tr.reset_index(drop=True),
            "val":   val.reset_index(drop=True)
        }
        print(f"[{cat}] {len(tr):,} train / {len(val):,} val samples")
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
    min_time = behaviors_df['Time'].min()
    max_time = behaviors_df['Time'].max()
    midpoint = min_time + (max_time - min_time) / 2
    print(f"Midpoint time computed: {midpoint}")
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

def init(process_dfs = False, process_behaviors = False, data_dir = 'dataset/train/', valid_data_dir = 'dataset/valid/', zip_file = f"MINDlarge_train.zip", valid_zip_file = f"MINDlarge_dev.zip", download=False,
    news_df_pkl="models/news_df_processed.pkl", train_df_pkl="models/train_df_processed.pkl", categorized_samples=False):
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
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    
    print("\nLoaded behaviors data:")
    print(behaviors_df.head())

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
        if "small" in data_dir:
            user_category_profiles_path = 'small_user_category_profiles.pkl'
            behaviors_df_processed_path = "small_behaviors_df_processed.pkl"
        else:
            user_category_profiles_path = 'user_category_profiles.pkl'
            behaviors_df_processed_path = "behaviors_df_processed.pkl"
        
        filtered_user_category_profiles.to_pickle(user_category_profiles_path)
        user_category_profiles = filtered_user_category_profiles
        print(f"\nSaved user_category_profiles to {user_category_profiles_path}")
        behaviors_df.to_pickle(behaviors_df_processed_path)
        print(f"\nSaved behaviors_df to {behaviors_df_processed_path}")
    else:
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
        user_category_profiles = pd.read_pickle("models/user_category_profiles.pkl")
        behaviors_df = pd.read_pickle("models/behaviors_df_processed.pkl")
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
        clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
            data_dir=data_dir,
            news_file=news_file,
            behaviors_file=behaviors_file,
            user_category_profiles=user_category_profiles,
            num_clusters=num_clusters,
            fraction=1,
            max_title_length=30,
            max_history_length=50,
            train_df_pkl=train_df_pkl, news_df_pkl=news_df_pkl, categorized_samples=categorized_samples
        )
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

    clustered_data = make_clustered_data(train_df, num_clusters)
    if categorized_samples:
        category_data = make_category_data(train_df)
    else:
        category_data = {}

    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    max_history_length = 50
    max_title_length = 30
    batch_size = 64
    return data_dir, vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters, category_data

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

def load_category_models(category_train_dfs, dataset_size='large', model_size='large', epochs=3, load_best_model=True, tokenizer=None, max_history_length=50, max_title_length=30, train_new=False, batch_size=256):
    model_dir = "models"
    category_models = {}
    for category, df in category_train_dfs.items():
        keras_model_save_path = f'{model_dir}/fastformer_{model_size}_category_{category}_{epochs}epochs.keras'
        model_save_path = f'{model_dir}/fastformer_{model_size}_category_{category}_{epochs}epochs.h5'
        best_model = f'{model_dir}/best_model_fastformer_{model_size}_category_{category}_{epochs}epochs.h5'
        keras_best_model = f'{model_dir}/fastformer_{model_size}_category_{category}_{epochs}epochs.keras'
        if load_best_model and os.path.exists(best_model):
            category_models[f"{category}_{model_size}"] = load(best_model)
        elif not load_best_model and os.path.exists(model_save_path):
            category_models[f"{category}_{model_size}"] = load(model_save_path)
        
    if category_models == {} or train_new:
        vocab_size = len(tokenizer.word_index) + 1
        category_models = train_category_models(category_train_dfs, vocab_size, max_history_length, max_title_length, batch_size=batch_size, epochs=epochs, dataset_size=dataset_size, train_only_new=False, train_fraction=0.9, load_best_model=True)
    return category_models

def log_print(s):
    logging.info(s)
    print(s)

def get_models(process_dfs = False, process_behaviors = False, data_dir = 'dataset/train/', valid_data_dir = 'dataset/valid/', zip_file = f"MINDlarge_train.zip", valid_zip_file = f"MINDlarge_dev.zip",
    model_type='cluster', dataset_size='large', model_size='large', load_best_model=False, load_best_models=[], epochs=1, retrain_models = [], evaluate=False, skip_already_evaluated=False, dataset_fraction=1.0,
    dataset='train', batch_size=256, eval_dataset_size='large'):
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'
    if "small" in data_dir:
        news_df_pkl = "models/small_news_df_processed"
        train_df_pkl = "models/small_train_df_processed"
    else:
        news_df_pkl = "models/news_df_processed"
        train_df_pkl = "models/train_df_processed"
    categorized_samples = False
    if model_type == 'category' or model_type == 'all':
        news_df_pkl = f"{news_df_pkl}_categorized"
        train_df_pkl = f"{train_df_pkl}_categorized"
        categorized_samples = True
    news_df_pkl = f"{news_df_pkl}.pkl"
    train_df_pkl = f"{train_df_pkl}.pkl"
    data_dir, vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters, category_data = init(process_dfs, process_behaviors, data_dir, valid_data_dir, zip_file, valid_zip_file,
    train_df_pkl=train_df_pkl, news_df_pkl=news_df_pkl, categorized_samples=categorized_samples)
    if process_dfs:
        clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
            data_dir=data_dir,
            news_file=news_file,
            behaviors_file=behaviors_file,
            user_category_profiles=user_category_profiles,
            num_clusters=3,
            fraction=1,
            max_title_length=30,
            max_history_length=50,
            train_df_pkl=train_df_pkl, news_df_pkl=news_df_pkl, categorized_samples=categorized_samples
        )
    quick_compare(behaviors_df, train_df)
    if model_type == 'cluster' or model_type == 'all':
        cluster_models = train_cluster_models(
            clustered_data=clustered_data,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            max_history_length=max_history_length,
            max_title_length=max_title_length,
            num_clusters=num_clusters,
            batch_size=batch_size,
            epochs=1,
            size=model_size,
            retrain='cluster' in retrain_models
        )
        logging.info(f"loaded total cluster_models:{cluster_models}")
    if model_type == 'global' or model_type == 'all':
        global_model = train_global_model(train_df, tokenizer, vocab_size, max_history_length, max_title_length, dataset_size=model_size, batch_size=batch_size, epochs=epochs, retrain='global' in retrain_models, load_best_model='global' in load_best_models)
        global_models = {}
        global_models[f"global_{model_size}"] = global_model
        logging.info(f"loaded total global_models:{global_models}")
    if model_type == 'category' or model_type == 'all':
        #category_train_dfs, news_df, behaviors_df, tokenizer = prepare_category_train_dfs(data_dir, news_file, behaviors_file, 30, 50, f"category_train_dfs_{dataset_size}.pkl")
        category_train_dfs = train_df
        category_models = load_category_models(category_data, dataset_size=dataset_size, model_size=model_size, load_best_model=load_best_model or 'category' in load_best_models,
        epochs=epochs, tokenizer=tokenizer, train_new='category' in retrain_models, batch_size=batch_size)
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
        for key, model in models.items():
            store_path = out_dir / f"{key}_{dataset}_{dataset_size}.parquet"
            store_metrics_path = out_dir / f"{key}_{dataset}_{dataset_size}_metrics.json"
            if skip_already_evaluated and store_path.exists() and store_path.stat().st_size > 0:
                log_print(f"Skipping evaluation for {key}: {store_path}")
                continue
            res = evaluate_with_generator(
                model,
                eval_df=train_df,
                batch_size=batch_size,
                store_path=store_path,
                flush_every=0.10,
                verbose=1,
                store_metrics_path=store_metrics_path
            )
            logging.info(f"model {key} eval res:{res}")
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
        tree_method = 'default'):

    files = sorted(Path(parquet_dir).glob(pattern))
    if base_model_type == "cluster":
        keep = re.compile(r"^(?:\d+|global)_")
        files = [fp for fp in files if keep.match(fp.stem)]
    if base_model_type == "category":
        remove = re.compile(r"^(?:\d+)_")
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

def evaluate_meta_from_parquet(meta_path: str,
                                parquet_dir: str = "base_preds",
                                pattern: str = "*.parquet",
                                base_model_type = "all",
                                store_parquet_path="preds.parquet",
                                store_metrics_path="metrics.json",
                                verbose: bool = True,
                                booster='default'):

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
        keep = re.compile(r"^(?:\d+|global)_")
        files = [fp for fp in files if keep.match(fp.stem)]
    if base_model_type == "category":
        remove = re.compile(r"^(?:\d+)_")
        files = [fp for fp in files if not remove.match(fp.stem)]
    if not files:
        raise FileNotFoundError(f"No parquet files matching {pattern}")

    print(f"base files:{files}")
    merged   = None
    featcols = []

    for fp in files:
        df  = pd.read_parquet(fp)
        col = f"pred_{fp.stem}"
        df  = df.rename(columns={"y_pred": col})
        featcols.append(col)

        key_cols = ["row_id", "y_true"] if "row_id" in df.columns else ["y_true"]
        merged   = df[key_cols + [col]] if merged is None \
                   else merged.merge(df[key_cols + [col]], on=key_cols, how="inner")

    if "row_id" not in merged.columns:
        merged.insert(0, "row_id", np.arange(len(merged), dtype="int64"))
    print(f"featcols:{featcols}")
    X      = merged[featcols].to_numpy(dtype="float32")
    y_true = merged["y_true"].to_numpy(dtype="int8")

    y_pred = meta.predict_proba(X)[:, 1]
    merged["y_pred"] = y_pred.astype("float32")

    auc  = roc_auc_score(y_true, y_pred)
    ap   = average_precision_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, (y_pred >= 0.5).astype(int), average="binary", zero_division=0)

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

        cols_to_write = ["row_id", "y_true", "y_pred"]
        merged[cols_to_write].to_parquet(
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
            if self.num_users == 0:     # guard against div/0
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

def evaluate_parquet_scores(parquet_path, k_values = (5, 10, 20, 50)):
    # Computes ranking metrics from a saved .parquet prediction file.
    # The file must row_id, y_true and y_pred columns

    df = pd.read_parquet(parquet_path)

    # If grouped by user
    group_key = "user_id" if "user_id" in df.columns else None

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
from pathlib import Path

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

def main(dataset='train', process_dfs=False, process_behaviors=False,
        data_dir_train='dataset/train/', data_dir_valid='dataset/valid/',
        zip_file_train="MINDlarge_train.zip", zip_file_valid="MINDlarge_dev.zip",
        user_category_profiles_path='', user_cluster_df_path='', cluster_id=None, meta_train=False,resume=True,
        model_type="cluster", dataset_size='large', load_best_model=False, load_best_models=[], eval_scope='cluster', model_size='large', epochs=1,
        adaptivity_test=False, donor_strategy='random', shuffle=False, drift_fraction=0.5, use_full_set=True, eval_separate=False,
        skip_already_evaluated=False, batch_size=256, retrain_models=[], eval_dataset_size='large',
        ext_data_dir_train='dataset/train/', ext_data_dir_valid='dataset/valid/',
        ext_zip_file_train="MINDlarge_train.zip", ext_zip_file_valid="MINDlarge_dev.zip",):
    # Main function to run tests on a given dataset type ('train' or 'valid').
    # It uses the midpoint time as cutoff and then runs evaluations.
    # Choose dataset directory based on parameter.
    
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
    # Format the time to ISO format with a trailing 'Z'
    cutoff_time_str = midpoint_time.isoformat().replace('+00:00', 'Z')
    print("Using cutoff time:", cutoff_time_str)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(news_df["CombinedText"])
    
    full_data = ext_behaviors_df
    # train_test_split function gets train/test splits from behaviors_df.
    if use_full_set:
        if dataset.lower() == 'train':
            train_data = behaviors_df
            test_data = {}
        elif dataset.lower() == 'valid':
            test_data = behaviors_df
            train_data = {}
    else:
        train_data, test_data = train_test_split_time(behaviors_df, cutoff_time_str)

    if model_type == "cluster":
        models_dict, news_df, behaviors_df, tokenizer = get_models(process_dfs, process_behaviors, data_dir_train, data_dir_valid, zip_file_train, zip_file_valid, evaluate=eval_separate, dataset=dataset,
        skip_already_evaluated=skip_already_evaluated, model_size=model_size, batch_size=batch_size, retrain_models=retrain_models, dataset_size=dataset_size)
    if model_type == "category" or model_type == "all" or model_type == "global":
        models_dict, news_df, behaviors_df, tokenizer = get_models(process_dfs, process_behaviors, data_dir_train, data_dir_valid, zip_file_train, zip_file_valid, model_type=model_type, dataset_size=dataset_size,
        model_size=model_size, load_best_model=load_best_model, load_best_models=load_best_models, epochs=epochs, evaluate=eval_separate, dataset=dataset,
        skip_already_evaluated=skip_already_evaluated, batch_size=batch_size, retrain_models=retrain_models)

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

    meta_name = 'XGBClassifier_hist'
    meta_model_base = f"meta_model_{meta_name}_{model_type}_{model_size}_train_{dataset_size}"
    meta_model_base_pattern = f"meta_model_{meta_name}_*_{model_size}_train_{dataset_size}"
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
    if os.path.exists(filename):
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
    
    tensors_file = f"{dataset}_{dataset_size}_title_tensors.pkl"
    if not os.path.exists(tensors_file):
        build_and_save_title_tensors(
            news_df=news_df,
            tokenizer=tokenizer,
            max_title_length=30,
            out_path=tensors_file
        )

    with open(tensors_file, "rb") as f:
        data = pickle.load(f)
    padded = data["padded"]
    title_tensor = tf.convert_to_tensor(padded, dtype=tf.int32)
    id_to_index = data["id_to_index"]
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
        meta_model_pattern=meta_model_pattern
    )
    
    print("Evaluation complete. Intermediate results were written during testing.")
    return results_user_level, cluster_results_df
