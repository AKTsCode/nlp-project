# nlp-project

Important Note on Code Changes and Generated Files:

This project saves several intermediate files (like .pkl and .npz files) to speed up execution.

If you modify the code in a way that affects how these files are created or their content (e.g., changes to tokenization, vocabulary, IDF, or TF-IDF matrix generation), you must manually delete the affected files before running the code again.

Otherwise, the script might load outdated data, leading to incorrect results.

Files to delete:

    movie_metadata.pkl
    summaries.pkl
    idf_dict.pkl
    idf_dict.json
    tf_idf_sparse_matrix.npz
    word_counts.txt
    word_counts_sorted.txt

Deleting these files ensures the script regenerates them with your latest changes.