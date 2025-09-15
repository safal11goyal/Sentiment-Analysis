from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def vectorize_text(train_texts, test_texts=None, max_features=10000, ngram_range=(1,2), save_path=None):
    """
    Fit TF-IDF on training data and transform test data.
    """
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts) if test_texts is not None else None

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(tfidf, f)

    return X_train, X_test, tfidf
