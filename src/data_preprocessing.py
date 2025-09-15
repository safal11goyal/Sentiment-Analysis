import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text: str, use_stemming=False, use_lemmatization=True) -> str:
    """
    Clean and normalize raw text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)   # remove HTML
    text = re.sub(r"http\\S+|www\\S+", " ", text)  # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punct
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]

    if use_stemming:
        tokens = [stemmer.stem(w) for w in tokens]
    if use_lemmatization:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

def preprocess_dataset(df: pd.DataFrame, text_col: str, label_col: str = None) -> pd.DataFrame:
    """
    Apply text cleaning to a dataframe column.
    """
    df["clean_review"] = df[text_col].apply(clean_text)
    if label_col:
        return df[[text_col, "clean_review", label_col]]
    return df[[text_col, "clean_review"]]
