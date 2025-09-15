# Sentiment Analysis Project

A complete machine learning pipeline for sentiment analysis of movie reviews using the IMDB dataset. This project implements both baseline and advanced models to classify movie reviews as positive or negative.

## 📁 Project Structure

```
sentiment_analysis/
├── data/
│   ├── raw/                    # Raw dataset files
│   │   └── IMDB Dataset.csv
│   └── processed/              # Processed and cleaned data
│       ├── train.csv
│       ├── test.csv
│       ├── train_clean.csv
│       └── test_clean.csv
├── models/                     # Trained models and vectorizers
│   ├── baseline_model.pkl
│   ├── advanced_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── preprocessing.ipynb     # Data preprocessing
│   ├── model_baseline.ipynb    # Baseline model training
│   └── model_advanced.ipynb    # Advanced model training
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py   # Text cleaning functions
│   ├── feature_engineering.py  # TF-IDF vectorization
│   ├── train_model.py         # Model training functions
│   ├── evaluate.py            # Model evaluation
│   ├── predict.py             # Prediction interface
│   └── split_dataset.py       # Dataset splitting utility
├── main.py                     # Main pipeline script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd sentiment_analysis

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Place your IMDB dataset file as `data/raw/IMDB Dataset.csv`. The dataset should have two columns:
- `review`: Text of the movie review
- `sentiment`: Label (positive/negative)

### 3. Run the Complete Pipeline

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Train both baseline (Naive Bayes) and advanced (SVM) models
- Evaluate model performance
- Save trained models

### 4. Make Predictions

```bash
cd src
python predict.py
```

This starts an interactive prediction interface where you can input movie reviews and get sentiment predictions.

## 📊 Models

### Baseline Model: Naive Bayes
- **Algorithm**: Multinomial Naive Bayes
- **Features**: TF-IDF vectors (5000 features, 1-2 grams)
- **Performance**: ~86.5% accuracy

### Advanced Model: Support Vector Machine
- **Algorithm**: Linear SVM
- **Features**: TF-IDF vectors (10000 features, 1-2 grams)
- **Performance**: ~88.7% accuracy

## 🔧 Key Components

### Data Preprocessing (`src/data_preprocessing.py`)
- Text cleaning (HTML removal, punctuation, lowercase)
- Stop word removal
- Lemmatization/Stemming
- Tokenization

### Feature Engineering (`src/feature_engineering.py`)
- TF-IDF vectorization
- N-gram extraction (unigrams and bigrams)
- Feature selection and dimensionality control

### Model Training (`src/train_model.py`)
- Baseline model: Multinomial Naive Bayes
- Advanced models: Logistic Regression, Linear SVM
- Model persistence with pickle

### Evaluation (`src/evaluate.py`)
- Accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Classification reports

## 📈 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 86.52% | 0.87 | 0.87 | 0.87 |
| Linear SVM | 88.74% | 0.89 | 0.89 | 0.89 |

## 🛠️ Usage Examples

### Using the Main Pipeline
```python
from src.data_preprocessing import preprocess_dataset
from src.feature_engineering import vectorize_text
from src.train_model import train_advanced

# Load and preprocess data
df_clean = preprocess_dataset(df, "review", "sentiment")

# Create features
X_train, X_test, vectorizer = vectorize_text(train_texts, test_texts)

# Train model
model = train_advanced(X_train, y_train, model_type="svm")
```

### Making Predictions
```python
from src.predict import predict_review

# Single prediction
sentiment = predict_review("This movie was fantastic!")
print(f"Sentiment: {sentiment}")  # Output: positive
```

### Running Notebooks
The project includes several Jupyter notebooks for detailed analysis:

1. **EDA.ipynb**: Exploratory data analysis with visualizations
2. **preprocessing.ipynb**: Step-by-step text preprocessing
3. **model_baseline.ipynb**: Baseline model development
4. **model_advanced.ipynb**: Advanced model comparison

## 📋 Requirements

- Python 3.7+
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn
- jupyter

## 🔍 Features

- **Complete ML Pipeline**: From raw data to deployed model
- **Multiple Models**: Compare baseline vs advanced approaches
- **Interactive Prediction**: Real-time sentiment analysis
- **Comprehensive Evaluation**: Detailed performance metrics
- **Modular Design**: Reusable components
- **Jupyter Notebooks**: Step-by-step analysis and experimentation

## 🎯 Model Performance

The SVM model achieves the best performance with:
- **88.74% accuracy** on test data
- Balanced precision and recall for both classes
- Robust performance on various review types

## 🚀 Future Improvements

- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add cross-validation for better model selection
- [ ] Create web interface for predictions
- [ ] Add support for multi-class sentiment (neutral, very positive, etc.)
- [ ] Implement model versioning and experiment tracking
- [ ] Add data drift detection

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues, please open an issue in the repository or contact the maintainer.
