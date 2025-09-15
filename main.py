import pandas as pd
from src.data_preprocessing import preprocess_dataset
from src.feature_engineering import vectorize_text
from src.train_model import train_baseline, train_advanced
from src.evaluate import evaluate_model
import os

def main():
    # =============================
    # 1. Load Raw Data
    # =============================
    raw_path = "data/raw/IMDB Dataset.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset not found at {raw_path}")
    
    df = pd.read_csv(raw_path)
    print(f"✅ Loaded dataset with {len(df)} rows")

    # =============================
    # 2. Train/Test Split
    # =============================
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # =============================
    # 3. Preprocess Text
    # =============================
    train_clean = preprocess_dataset(train_df, "review", "sentiment")
    test_clean = preprocess_dataset(test_df, "review", "sentiment")

    os.makedirs("data/processed", exist_ok=True)
    train_clean.to_csv("data/processed/train_clean.csv", index=False)
    test_clean.to_csv("data/processed/test_clean.csv", index=False)
    print("✅ Data preprocessing complete")

    # =============================
    # 4. Feature Engineering
    # =============================
    X_train, X_test, tfidf = vectorize_text(
        train_clean["clean_review"],
        test_clean["clean_review"],
        save_path="models/tfidf_vectorizer.pkl"
    )
    y_train, y_test = train_clean["sentiment"], test_clean["sentiment"]
    print("✅ TF-IDF vectorization complete")

    # =============================
    # 5. Train Models
    # =============================
    baseline = train_baseline(X_train, y_train, save_path="models/baseline_model.pkl")
    svm = train_advanced(X_train, y_train, model_type="svm", save_path="models/advanced_model.pkl")
    print("✅ Models trained and saved")

    # =============================
    # 6. Evaluate Models
    # =============================
    print("\n📊 Baseline Model (Naive Bayes):")
    acc_base = evaluate_model(baseline, X_test, y_test, "Naive Bayes")

    print("\n📊 Advanced Model (SVM):")
    acc_svm = evaluate_model(svm, X_test, y_test, "SVM")

    # =============================
    # 7. Choose Best Model
    # =============================
    if acc_svm >= acc_base:
        print(f"\n✅ Best model: SVM ({acc_svm:.4f})")
    else:
        print(f"\n✅ Best model: Naive Bayes ({acc_base:.4f})")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
