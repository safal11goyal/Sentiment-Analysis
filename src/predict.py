import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from src.data_preprocessing import clean_text

def load_model(model_path="../models/advanced_model.pkl", vectorizer_path="../models/tfidf_vectorizer.pkl"):
    """Load trained model and TF-IDF vectorizer."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_review(review: str, model=None, vectorizer=None):
    """Predict sentiment of a single review."""
    if model is None or vectorizer is None:
        model, vectorizer = load_model()
    clean = clean_text(review)
    review_vec = vectorizer.transform([clean])
    prediction = model.predict(review_vec)[0]
    return prediction

if __name__ == "__main__":
    print("📝 Sentiment Analysis Prediction (type 'exit' to quit)")
    model, vectorizer = load_model()
    
    while True:
        review = input("\nEnter a review: ")
        if review.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        sentiment = predict_review(review, model, vectorizer)
        print(f"Predicted Sentiment: {sentiment}")
