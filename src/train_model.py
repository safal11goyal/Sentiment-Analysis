import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def train_baseline(X_train, y_train, save_path="../models/baseline_model.pkl"):
    """
    Train Naive Bayes baseline model.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)

    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    return model

def train_advanced(X_train, y_train, model_type="svm", save_path="../models/advanced_model.pkl"):
    """
    Train Logistic Regression or SVM.
    """
    if model_type == "logreg":
        model = LogisticRegression(max_iter=200)
    else:
        model = LinearSVC()

    model.fit(X_train, y_train)

    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    return model
