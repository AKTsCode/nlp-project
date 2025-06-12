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

    # removing stop words
    tokens = [t for t in tokens if t not in stop_words]

    return tokens
    


# create vocabulary -> delete duplicate words
def createVocabulary(words: List[str]) -> List[str]:
    vocabulary = sorted(set(words))
    return list(vocabulary)
    

# read plot_summaries.txt
with open ("raw/plot_summaries.txt", "r", encoding="utf-8") as file:
    data = file.read()


# tokenization = list of all words in the corpus
corpus_tokens = tokenzie(data)

# vocabulary
vocabulary = createVocabulary(corpus_tokens)


print("Vocabulary direkt nach Erstellung:")
for i, token in enumerate(vocabulary[:50]):
    print(f"{i}: {token}")

# without stopwords =  192454 words in vocabulary
# with stopwords = 192592 (138 Stopwords)

print("Vocabulary Length:")
print(len(vocabulary))


# number of each word in corpus_tokens
word_counts = Counter(corpus_tokens)

with open("word_counts.txt", "w", encoding="utf-8") as f:
    for word, count in word_counts.items():
        f.write(f"{word}: {count}\n")

with open("word_counts_sorted.txt", "w", encoding="utf-8") as f:
    for word, count in word_counts.most_common():
        f.write(f"{word}: {count}\n")



# movie-metadata dictionary: {   "123": {     "title": "Inception",     "genres": ["Action", "Sci-Fi"]   } }
movie_metadata_path = "movie_metadata.pkl"
movie_metadata = {}

if os.path.exists(movie_metadata_path):
    with open(movie_metadata_path, "rb") as f:
        movie_metadata = pickle.load(f)
    print("movie_metadata loaded")
else:
    with open ("raw/movie.metadata.tsv", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: # not enough info
                continue 

            movie_id = parts[0]
            title = parts[2] if len(parts) > 2 else ""
            genres_raw = parts[-1] # last field

            try:
                genres = list(json.loads(genres_raw).values())
            except:
                genres = []
            movie_metadata[movie_id] = {
                "title": title,
                "genres": genres
            }
    with open(movie_metadata_path, "wb") as f:
        pickle.dump(movie_metadata, f)
    print("movie_metadata created")



# create list of summaries
@dataclass
class Summary:
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
    # get summaries if already created
    with open("summaries.pkl", "rb") as f:
        summaries = pickle.load(f)
    print("summaries.pkl loaded.")
else:
    with open ("raw/plot_summaries.txt", "r", encoding="utf-8") as f:
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
    # safe summaries
    with open("summaries.pkl", "wb") as f:
        pickle.dump(summaries, f)
    print("summaries.pkl created.")

print("Anzahl an Summaries nach Erstellung")
print(len(summaries))

print("Vocabulary Length:")
print(len(vocabulary))

# IDF values for all tokens

idf_path = "idf_dict.pkl"
idf_dict: dict[str, float] = {}

if os.path.exists(idf_path):
    with open (idf_path, "rb") as f:
        idf_dict = pickle.load(f)
    print("idf_dict loaded")
else:   
    print("Vocabulary kurz vor IDF berechnungen:")
    for i, token in enumerate(vocabulary[:50]):
        print(f"{i}: {token}")
    print("Vocabulary Length:")
    print(len(vocabulary))

    n = len(summaries)
    for token in vocabulary:
        # df = Document Frequency -> in how many summaries does the token appear?
        df = sum(1 for s in tqdm(summaries, desc=token) if token in s.vocabulary)
        
        # idf = Inverse Document Frequency --> how rare is the token in the corpus
        idf = math.log(n / df)

        idf_dict[token] = idf
    with open(idf_path, "wb") as f:
        pickle.dump(idf_dict, f)
    # safe as json. maybe needed later
    with open("idf_dict.json", "w", encoding="utf-8") as f:
        idf_dict_rounded = {k: round(v, 4) for k, v in idf_dict.items()}
        json.dump(idf_dict_rounded, f, indent=2)        
#print(idf_dict)


# TF-IDF Vektor Dictionary
# { 
#   "32131": {                      -> Summary ID
#              "cat": 0.003         -> TF-IDF Values for each word from vocabulary
#              "dog": 0.004,
#              "lasersword": 0.0}                 
#   ....
#


# ---
## Creation and storage of the TF-IDF sparse matrix
# ---


# Vokabular-Mapping Template
vocabulary_to_idx = {word: i for i, word in enumerate(vocabulary)}
num_docs = len(summaries)
num_vocab = len(vocabulary)


# create empty Lil-Matrix (List-of-Lists)
# (Number of docs, vocabulary-size)
tf_idf_sparse_matrix_path = "tf_idf_sparse_matrix.npz"
tf_idf_sparse_matrix = None

if os.path.exists(tf_idf_sparse_matrix_path):
    tf_idf_sparse_matrix = load_npz(tf_idf_sparse_matrix_path)
    print("tf_idf_sparse_matrix loaded.")
else:
    print("Starting creation of the TF-IDF sparse matrix...")
    tf_idf_sparse_matrix = lil_matrix((num_docs, num_vocab), dtype=np.float32)

    #run through all summaries and fill the sparse matrix
    # doc_idx = row index
    for doc_idx, s in tqdm(enumerate(summaries), total=num_docs, desc="Create TF-IDF Matrix"):
        # iterate through s.term_freqs
        for word, tf in s.term_freqs.items():
            # get the index of the word in vocabulary
            word_idx = vocabulary_to_idx[word]
            idf = idf_dict.get(word, 0)
            tf_idf = np.float32(tf * idf)

            if tf_idf > 0:
                tf_idf_sparse_matrix[doc_idx, word_idx] = tf_idf

    # Conert the LIL-Matrix in a CSR-Matrix (Compressed Sparse Row)
    # more efficient for mathematical operations (matrix * vector)
    tf_idf_sparse_matrix = tf_idf_sparse_matrix.tocsr()

    save_npz(tf_idf_sparse_matrix_path, tf_idf_sparse_matrix)
    print(f"TF-IDF Sparse Matrix created")

# Optional: Überprüfe die Dimensionen der erstellten Matrix
print(f"Dimension der TF-IDF Sparse Matrix: {tf_idf_sparse_matrix.shape}")



























# tf_idf_vector_dict_path = "tf_idf_vector_dict.pkl"
#
# tf_idf_vector_dict: dict[str , dict[str, float]] = {}
#
# if os.path.exists(tf_idf_vector_dict_path):
#     with open (tf_idf_vector_dict_path, "rb") as f:
#         tf_idf_vector_dict = pickle.load(f)
#     print("tf_idf_vector_dict loaded")
# else:
#
#     print("Anzahl an Summaries vor tf-idf vektor Erstellung")
#     print(len(summaries))
#
#     for s in summaries:
#         doc_id = s.id
#         # with float -> ~70 GB
#         # with np.float32 -> ~35 GB
#         tf_idf_vector: dict[str, np.float32] = {}
#
#
#         for word in vocabulary:
#             tf = s.term_freqs.get(word, 0)
#             idf = idf_dict.get(word, 0)
#             # with float -> ~70 GB
#             # with np.float32 -> ~35 GB
#             tf_idf = np.float32(tf*idf)
#             tf_idf_vector[word] = tf_idf
#
#         tf_idf_vector_dict[doc_id] = tf_idf_vector
#     with open(tf_idf_vector_dict_path, "wb") as f:
#         pickle.dump(tf_idf_vector_dict, f)
#
# # show the first 5 TF_IDF vectors
# for i, (doc_id, tf_idf_vector) in enumerate(tf_idf_vector_dict.items()):
#     if i >= 5:
#         break
#
#     summary = next((s for s in summaries if s.id == doc_id), None)
#     if summary:
#         print(f"\n🎬 Title: {summary.title}")
#         print(f"📁 Genres: {', '.join(summary.genres)}")
#         print(f"🆔 ID: {doc_id}")
#         print("🔢 Top TF-IDF values (sorted):")
#         for word, score in sorted(tf_idf_vector.items(), key=lambda item: item[1], reverse=True):
#             print(f"  {word}: {score:.4f}")