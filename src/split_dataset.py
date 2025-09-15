import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================
# 1. Load dataset
# ==========================
# Change the path to your dataset
data_path = "../data/raw/IMDB Dataset.csv"
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print(df.head())

# ==========================
# 2. Split into Train/Test
# ==========================
# 80% train, 20% test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['sentiment'],  # ensures balanced distribution of labels
    random_state=42
)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ==========================
# 3. Save to processed folder
# ==========================
train_df.to_csv("../data/processed/train.csv", index=False)
test_df.to_csv("../data/processed/test.csv", index=False)

print("✅ Dataset split completed!")
print("Train and test files saved in ../data/processed/")
