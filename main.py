from typing import List
import os
import re
import string
from collections import Counter
from dataclasses import dataclass
import json
import pickle
import math
import nltk
import numpy as np
from scipy.sparse import lil_matrix, save_npz, load_npz
from typing import Set
from tqdm import tqdm #pip install tqdm
from nltk.corpus import stopwords #pip install nltk

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

import joblib

# ==============================================================================
# --- Text Preprocessing Functions ---
# ==============================================================================

# Download stopwords if not available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# create list of all words
def tokenzie(text: str) -> List[str]:
    # remove numbers
    text = re.sub(r"\d+","",text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # make big letters small
    text = text.lower()

    # creating list
    tokens = text.split()

    # filter out stop words
    tokens = [t for t in tokens if t not in stop_words]

    return tokens
    


#  Function: delete duplicate words -> get sorted list of unique words
def createVocabulary(words: List[str]) -> List[str]:
    vocabulary = sorted(set(words))
    return list(vocabulary)




# ==============================================================================
# --- Corpus Tokenization and Vocabulary Creation ---
# ==============================================================================

print("--- Starting Corpus Tokenization and Vocabulary Creation ---")

# Read plot_summaries.txt to get the raw corpus text
with open ("raw/plot_summaries.txt", "r", encoding="utf-8") as file:
    data = file.read()


# Tokenize the entire corpus to get a list of all words
corpus_tokens = tokenzie(data)

# Create sorted vocabulary from the corpus tokens
vocabulary = createVocabulary(corpus_tokens)


print(f"Vocabulary Length after creating: {len(vocabulary)} words")
print("First 50 words of the Vocabulary:")
for i, token in enumerate(vocabulary[:50]):
    print(f"  {i}: {token}")

# without stopwords =  192454 words in vocabulary
# with stopwords = 192592 (138 Stopwords)

# Count word frequencies across the entire corpus
word_counts = Counter(corpus_tokens)

# Save word counts to a text file (unsorted)
with open("word_counts.txt", "w", encoding="utf-8") as f:
    for word, count in word_counts.items():
        f.write(f"{word}: {count}\n")

# Save word counts sorted by frequency (most common first)
with open("word_counts_sorted.txt", "w", encoding="utf-8") as f:
    for word, count in word_counts.most_common():
        f.write(f"{word}: {count}\n")

print("--- Corpus Tokenization and Vocabulary Creation Complete ---")




# ==============================================================================
# --- Movie Metadata Loading / Creation ---
# ==============================================================================

print("\n--- Starting Movie Metadata Processing ---")

# movie-metadata dictionary: {   "123": {     "title": "Inception",     "genres": ["Action", "Sci-Fi"]   } }
movie_metadata_path = "movie_metadata.pkl"
movie_metadata = {}

if os.path.exists(movie_metadata_path):
    with open(movie_metadata_path, "rb") as f:
        movie_metadata = pickle.load(f)
    print("Movie metadata loaded from .pkl file.")
else:
    print("Movie metadata .pkl not found. Creating from .tsv...")
    with open ("raw/movie.metadata.tsv", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: # If not enough info
                continue

            movie_id = parts[0]
            title = parts[2] if len(parts) > 2 else "" # Title is typically at index 2
            genres_raw = parts[-1] # Genres are typically the last field

            try:
                genres = list(json.loads(genres_raw).values()) # not needed, optional
            except:
                genres = []
            movie_metadata[movie_id] = {
                "title": title,
                "genres": genres
            }
    with open(movie_metadata_path, "wb") as f:
        pickle.dump(movie_metadata, f)
    print("movie_metadata created")

print("--- Movie Metadata Processing Complete ---")




# ==============================================================================
# --- Summary List Creation ---
# ==============================================================================

print("\n--- Starting Summary List Processing ---")

@dataclass
class Summary:
    """
    Dataclass to hold all relevant information for a movie summary.
    """
    id: str
    title: str
    genres: List[str]
    text: str
    tokens: List[str]
    term_freqs: Counter
    vocabulary: Set[str]

summaries_path = "summaries.pkl"
summaries: List[Summary] = []

if os.path.exists(summaries_path):
    # Get summaries if already created
    with open("summaries.pkl", "rb") as f:
        summaries = pickle.load(f)
    print("Summaries loaded from .pkl file.")
else:
    with open ("raw/plot_summaries.txt", "r", encoding="utf-8") as f:
        print("Summaries .pkl not found. Creating from plot summaries and metadata...")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            id_ = parts[0]
            text = parts[1] if len(parts) > 1 else ""

            title = movie_metadata.get(id_, {}).get("title", "")
            genres = movie_metadata.get(id_, {}).get("genres", [])
            tokens = tokenzie(text)
            term_freqs = Counter(tokens)
            s_vocabulary = set(tokens)

            s = Summary(
                id = id_,
                title = title,
                genres = genres,
                text = text,
                tokens = tokens,
                term_freqs = term_freqs,
                vocabulary = s_vocabulary
            )
            summaries.append(s)
    # Save the complete list
    with open("summaries.pkl", "wb") as f:
        pickle.dump(summaries, f)
    print("summaries.pkl created.")

print(f"Total number of summaries processed: {len(summaries)}")
print("--- Summary List Processing Complete ---")




# ==============================================================================
# --- Inverse Document Frequency (IDF) Calculation ---
# ==============================================================================

print("\n--- Starting IDF Calculation ---")

idf_path = "idf_dict.pkl"
idf_dict: dict[str, float] = {}

if os.path.exists(idf_path):
    with open (idf_path, "rb") as f:
        idf_dict = pickle.load(f)
    print("IDF dictionary loaded from .pkl file.")
else:
    print("IDF dictionary .pkl not found. Calculating IDF values...")

    n = len(summaries) # Total number of summaries/docs
    for token in vocabulary:
        # Document Frequency (df): count in how many summaries the token appears
        df = sum(1 for s in tqdm(summaries, desc=token) if token in s.vocabulary)

        # Inverse Document Frequency: IDF = log(n / df)
        # IDF value shows how rare is the token in the corpus
        idf = math.log(n / df)

        idf_dict[token] = idf

    with open(idf_path, "wb") as f:
        pickle.dump(idf_dict, f)
    print("IDF dictionary created and saved to .pkl file.")

    # Optionally save as JSON (rounded for readability/debugging)
    with open("idf_dict.json", "w", encoding="utf-8") as f:
        idf_dict_rounded = {k: round(v, 4) for k, v in idf_dict.items()}
        json.dump(idf_dict_rounded, f, indent=2)

print("--- IDF Calculation Complete ---")



# ==============================================================================
# --- TF-IDF Sparse Matrix Creation and Storage ---
# ==============================================================================

print("\n--- Starting TF-IDF Sparse Matrix Processing ---")

# Create a mapping from vocabulary words to their column indices in the matrix
vocabulary_to_idx = {word: i for i, word in enumerate(vocabulary)}
num_docs = len(summaries)
num_vocab = len(vocabulary)


tf_idf_sparse_matrix_path = "tf_idf_sparse_matrix.npz"
tf_idf_sparse_matrix = None

if os.path.exists(tf_idf_sparse_matrix_path):
    # Loaded sparse matrix will be in CSR format
    tf_idf_sparse_matrix = load_npz(tf_idf_sparse_matrix_path)
    print("TF-IDF sparse matrix loaded from .npz file.")
else:
    print("TF-IDF sparse matrix .npz not found. Creating new matrix...")

    # Initialize a LIL (List of Lists) matrix for efficient incremental filling
    tf_idf_sparse_matrix = lil_matrix((num_docs, num_vocab), dtype=np.float32)

    # Iterate through each summary (row in the matrix)
    # doc_idx = row index
    for doc_idx, s in tqdm(enumerate(summaries), total=num_docs, desc="Create TF-IDF Matrix"):
        # Iterate through words that actually appear in the current summary
        for word, tf in s.term_freqs.items():
            # Get the index of the word from the global vocabulary map
            word_idx = vocabulary_to_idx[word]
            idf = idf_dict.get(word, 0)
            tf_idf = np.float32(tf * idf)

            if tf_idf > 0:
                tf_idf_sparse_matrix[doc_idx, word_idx] = tf_idf

    # Conert the LIL-Matrix to CSR (Compressed Sparse Row) Matrix format
    # more efficient for mathematical operations ( matrix-vector multiplication, ect.)
    tf_idf_sparse_matrix = tf_idf_sparse_matrix.tocsr()

    save_npz(tf_idf_sparse_matrix_path, tf_idf_sparse_matrix)
    print(f"TF-IDF Sparse Matrix created")

# Display the dimensions of the TF-IDF matrix
print(f"Dimension of TF-IDF Sparse Matrix: {tf_idf_sparse_matrix.shape}")
print("--- TF-IDF Sparse Matrix Processing Complete ---")



# ==============================================================================
# --- Prepare Labels (y) for Multi-Label Classification ---
# ==============================================================================

print("\n--- Preparing Label Matrix (y) ---")


# List of lists
# Outer list contains index of all summaries (row of our TD-IDF Matrix)
# Inner list contains genres for a summary
genre_labels = [s.genres for s in summaries]


# ===========================
# Filtering rare Genres


# Count the frequency for each genre
# Flatter the list
all_flat_genres_raw = [genre for sublist in genre_labels for genre in sublist]
genre_counts_raw = Counter(all_flat_genres_raw)

# Define threshold
# need to delete genre_classifier_model.pkl after changing
MIN_GENRE_FREQUENCY = 30

# Create a set of genres that meet the threshold
frequent_genres = {genre for genre, count in genre_counts_raw.items() if count >= MIN_GENRE_FREQUENCY}

print(f"Original number of unique genres: {len(genre_counts_raw)}")
print(f"Number of genres after filtering (min frequency >= {MIN_GENRE_FREQUENCY}): {len(frequent_genres)}")
print(f"Top 20 most common genres (before filtering): {genre_counts_raw.most_common(20)}")

# Filter the genre_labels lists: Keep only the genres considered 'frequent'
filtered_genre_labels = []
for movie_genres_list in genre_labels:
    # New list of genres for this movie, containing only the 'frequent_genres'
    current_movie_filtered_genres = [g for g in movie_genres_list if g in frequent_genres]
    # Add the filtered list. Important: A movie might end up with an empty list
    # if all its original genres were below the threshold. MultiLabelBinarizer can handle this.
    filtered_genre_labels.append(current_movie_filtered_genres)


# ===========================


# MultiLabelBinarizer
# This object will learn all unique genres present across all movies.
mlb = MultiLabelBinarizer()

# Transform the list of genre lists into a multi-hot-encoded (binary) matrix.
# `fit_transform` first learns the unique classes (genres) and then transforms the input.
# y matrix structure: rows = movie summaries, columns = unique genres, values = 1 or 0
y = mlb.fit_transform(filtered_genre_labels) # use genre_labels if no filtering wanted

# Save the MultiLabelBinarizer
# joblib.dump(mlb, mlb_path)
# print(f"MultiLabelBinarizer created and saved to {mlb_path}")

print(f"Shape of label matrix (y): {y.shape}")
# The shape should be (number of movies, number of unique genres).
print(f"Unique genres learned by MLB (column order in y): {list(mlb.classes_)}")

# Optional: Display the counts of each genre to understand genre distribution
all_flat_genres = [genre for sublist in filtered_genre_labels for genre in sublist]
genre_counts = Counter(all_flat_genres)
print(f"Total unique genres after filtering: {len(genre_counts)}")
print("Top 10 most common genres (after filtering):", genre_counts.most_common(10))


print("--- Label Matrix (y) Preparation Complete ---")



# ==============================================================================
# --- Model Training (One-vs-Rest Logistic Regression) ---
# ==============================================================================

print("\n--- Starting Model Training ---")

# Split data into training and testing sets
# X is your TF-IDF sparse matrix, y is your multi-hot-encoded genre matrix.
# test_size=0.2 means 20% of data will be used for testing, 80% for training.
# random_state ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(
    tf_idf_sparse_matrix, y, test_size=0.2, random_state=42
)
print(f"Shape of X_train (training features): {X_train.shape}")
print(f"Shape of X_test (testing features): {X_test.shape}")
print(f"Shape of y_train (training labels): {y_train.shape}")
print(f"Shape of y_test (testing labels): {y_test.shape}")

model_path = "genre_classifier_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Trained model loaded from .pkl file.")
else:

    # Initialize the OneVsRestClassifier model
    # This wrapper takes a base classifier (here, LogisticRegression) and trains one instance
    # of it for each target label (genre).
    # solver='liblinear' is a good choice for smaller datasets and sparse data.
    # max_iter increases the maximum number of iterations for convergence.
    base_classifier = LogisticRegression(solver='liblinear', max_iter=500, random_state=42)
    model = OneVsRestClassifier(base_classifier)

    # Train the model
    # The model learns the relationships between TF-IDF features and genres using the training data.
    print("Training the OneVsRest Logistic Regression model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Trained model created and saved to {model_path}")

# Make predictions on the test set
# predict() returns binary predictions (0 or 1 for each genre).
y_pred = model.predict(X_test)
# predict_proba() returns probabilities (between 0 and 1 for each genre).
y_proba = model.predict_proba(X_test)


# Evaluate model performance
# For multi-label classification, `accuracy_score` often reflects Jaccard similarity.
# `classification_report` provides more detailed metrics (precision, recall, F1-score) per label.
print("\n--- Model Evaluation ---")

# Evaluate overall performance using common multi-label metrics
# average='micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# average='macro': Calculate metrics for each label, and find their unweighted mean.
# average='samples': Calculate metrics for each sample, and find their average.
print(f"Micro F1-Score: {f1_score(y_test, y_pred, average='micro'):.4f}")
print(f"Macro F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Micro Precision: {precision_score(y_test, y_pred, average='micro'):.4f}")
print(f"Micro Recall: {recall_score(y_test, y_pred, average='micro'):.4f}")


# Without filtering rare genres or genres which dont exist in the test data:
# --- Model Evaluation ---
# Micro F1-Score: 0.3791
# Macro F1-Score: 0.0602
# C:\Users\akt20\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py:1706: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no true nor predicted samples. Use `zero_division` parameter to control this behavior.
# _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
# Micro Precision: 0.5028
# Micro Recall: 0.3042

# Threshold = 10
# # --- Model Evaluation ---
# Micro F1-Score: 0.3798
# Macro F1-Score: 0.0838
# C:\Users\akt20\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py:1706: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no true nor predicted samples. Use `zero_division` parameter to control this behavior.
# _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
# Micro Precision: 0.5028
# Micro Recall: 0.3051

print("\nDetailed Classification Report (per genre):\n")
# target_names are crucial here to see which genre corresponds to which row in the report.
target_names = mlb.classes_
print(classification_report(y_test, y_pred, target_names=target_names))

print("--- Model Training and Evaluation Complete ---")