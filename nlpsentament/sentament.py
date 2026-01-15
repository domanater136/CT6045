import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tracemalloc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, mutual_info_classif, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD
import umap


def load_clean_reviews(path):
    dataset = pd.read_csv(path)
    return dataset[['clean_review', 'polarity']]


def vectorize_text(dataset):
    X = dataset['clean_review']
    y = dataset['polarity']

    vectorizer = TfidfVectorizer(max_features = 10000, ngram_range=(1, 2)) # Include Bi-grams
    X_vec = vectorizer.fit_transform(X)

    return train_test_split(X_vec, y, test_size=0.2, random_state=6942067)


def run_naive_bayes(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)

    return predictions, report


def run_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=150)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)

    return predictions, report


def plot_accuracy(nb, lr):
    scores = {"Naive Bayes": nb['accuracy'], "Logistic Regression": lr['accuracy']}

    series = pd.Series(scores)
    ax = series.plot(kind='bar')
    ax.set_ylim(0.7, 1.0)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=0)  # Be readable please
    plt.title("Sentiment Accuracy Comparison")
    for idx, value in enumerate(series):
        ax.text(idx, value - 0.02, f"{value:.4f}", color='white', fontsize=12)
    plt.show()


def plot_closeness(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # USE DECIMALS >:(
    plt.title(title)
    plt.xlabel("predicited")
    plt.ylabel("actual")
    plt.show()

def plot_accuracy_multi(results_dict, ylabel):
    scores = {name: res['accuracy'] for name, res in results_dict.items()}
    series = pd.Series(scores)
    ax = series.plot(kind='bar', color=['blue','green','red'])
    plt.ylim(0.7, 1.0)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.title("Comparison of Sentiment Models")
    for idx, value in enumerate(series):
        ax.text(idx, value - 0.02, f"{value:.4f}", color='white', fontsize=12)
    plt.show()

def plot_time_memory(results_dict):
    # Time plot
    times = {name: res['time'] for name, res in results_dict.items()}
    series = pd.Series(times)
    ax = series.plot(kind='bar', color=['blue','green','red'])
    plt.ylabel("Seconds")
    plt.title("Time Taken by Each Model")
    plt.xticks(rotation=0)
    for idx, value in enumerate(series):
        ax.text(idx, value + 0.01, f"{value:.2f}", color='black', fontsize=10)
    plt.show()

    # Memory plot
    mems = {name: res['memory'] for name, res in results_dict.items()}
    series = pd.Series(mems)
    ax = series.plot(kind='bar', color=['skyblue','lightgreen','salmon'])
    plt.ylabel("Peak Memory (MB)")
    plt.title("Peak Memory Usage by Each Model")
    plt.xticks(rotation=0)
    for idx, value in enumerate(series):
        ax.text(idx, value + 0.1, f"{value:.2f}", color='black', fontsize=10)
    plt.show()

def run_sentiment_models(dataset):

    X_train, X_test, y_train, y_test = vectorize_text(dataset)

    start_time = time.time()
    nb_preds, nb_report = run_naive_bayes(X_train, X_test, y_train, y_test)
    fin_time = time.time() - start_time
    print(f"Naive Bayes took {fin_time} seconds.")
    print(f"Accuracy is {nb_report['accuracy']}")
    print("")

    start_time = time.time()
    lr_preds, lr_report = run_logistic_regression(X_train, X_test, y_train, y_test)
    fin_time = time.time() - start_time
    print(f"Logistic Regresion completed in {fin_time} seconds.")
    print(f"Accuracy is {lr_report['accuracy']}")
    print("")

    return {
        "nb_report": nb_report,
        "lr_report": lr_report,
        "y_test": y_test,
        "nb_preds": nb_preds,
        "lr_preds": lr_preds
    }


if __name__ == "__main__":
    dataset = load_clean_reviews("amazon_clean_nlp.csv")

    SENTAMENT = False
    if SENTAMENT:
        result = run_sentiment_models(dataset)
        plot_accuracy(result['nb_report'], result['lr_report'])
        plot_closeness(result['y_test'], result['nb_preds'], "Naive Bayes Closeness")
        plot_closeness(result['y_test'], result['lr_preds'], "Logistic Regression Closeness")

    # Testing reductions
    X_train, X_test, y_train, y_test = vectorize_text(dataset)

    results = {}
    results_time_mem = {}

    # Baseline NB
    print("NB")
    tracemalloc.start()
    start_time = time.time()
    nb_preds, nb_report = run_naive_bayes(X_train, X_test, y_train, y_test)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results['Baseline NB'] = {'accuracy': nb_report['accuracy'], 'y_pred': nb_preds}
    results_time_mem['Baseline NB'] = {'time': elapsed, 'memory': peak / 1e6}
    print(f"NB: {elapsed:.2f}s, {peak / 1e6:.2f} MB\n")

    # Baseline LR
    print("LR")
    tracemalloc.start()
    start_time = time.time()
    lr_preds, lr_report = run_logistic_regression(X_train, X_test, y_train, y_test)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results['Baseline LR'] = {'accuracy': lr_report['accuracy'], 'y_pred': lr_preds}
    results_time_mem['Baseline LR'] = {'time': elapsed, 'memory': peak / 1e6}
    print(f"LR: {elapsed:.2f}s, {peak / 1e6:.2f} MB\n")

    # Chi2
    print("Chi2 + LR")
    chi_selector = SelectKBest(chi2, k=2000)
    X_train_chi = chi_selector.fit_transform(X_train, y_train)
    X_test_chi = chi_selector.transform(X_test)

    tracemalloc.start()
    start_time = time.time()
    lr_chi_preds, lr_chi_report = run_logistic_regression(X_train_chi, X_test_chi, y_train, y_test)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results['Chi2+LR'] = {'accuracy': lr_chi_report['accuracy'], 'y_pred': lr_chi_preds}
    results_time_mem['Chi2+LR'] = {'time': elapsed, 'memory': peak / 1e6}
    print(f"Chi2+LR: {elapsed:.2f}s, {peak / 1e6:.2f} MB\n")

    # LR + Chi2 + SVD
    print("Chi2 + SVD + LR")
    svd = TruncatedSVD(n_components=300, random_state=6942067)
    X_train_chi_svd = svd.fit_transform(X_train_chi)
    X_test_chi_svd = svd.transform(X_test_chi)

    tracemalloc.start()
    start_time = time.time()
    lr_chi_svd_preds, lr_chi_svd_report = run_logistic_regression(X_train_chi_svd, X_test_chi_svd, y_train, y_test)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results['Chi2+SVD+LR'] = {'accuracy': lr_chi_svd_report['accuracy'], 'y_pred': lr_chi_svd_preds}
    results_time_mem['Chi2+SVD+LR'] = {'time': elapsed, 'memory': peak / 1e6}
    print(f"Chi2+SVD+LR: {elapsed:.2f}s, {peak / 1e6:.2f} MB\n")

    # LR + Chi2 + UMAP
    print("Chi2 + UMAP + LR")

    sample_size = 5000 # Avoid memory explosion of putting a 4gb of data into storage
    np.random.seed(6942067)
    sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)

    chi_selector2 = SelectKBest(chi2, k=3000)
    X_train_chi2 = chi_selector2.fit_transform(X_train, y_train)
    X_test_chi2 = chi_selector2.transform(X_test)

    X_train_sample = X_train_chi2[sample_indices]

    umap_reducer = umap.UMAP(n_components=50, n_neighbors=15, min_dist=0.1, metric='cosine', n_jobs=-1)
    umap_reducer.fit(X_train_sample)

    X_train_umap = umap_reducer.transform(X_train_chi2)
    X_test_umap = umap_reducer.transform(X_test_chi2)

    print(f"UMAP reduced features: {X_train_chi2.shape[1]} to {X_train_umap.shape[1]}")

    tracemalloc.start()
    start_time = time.time()
    lr_umap_preds, lr_umap_report = run_logistic_regression(X_train_umap, X_test_umap, y_train, y_test)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results['Chi2+UMAP+LR'] = {'accuracy': lr_umap_report['accuracy'], 'y_pred': lr_umap_preds}
    results_time_mem['Chi2+UMAP+LR'] = {'time': elapsed, 'memory': peak / 1e6}
    print(f"Chi2+UMAP+LR: {elapsed:.2f}s, {peak / 1e6:.2f} MB")
    print(f"Accuracy: {lr_umap_report['accuracy']:.4f}\n")

    # MI + LR
    print("MI + LR")
    mi_selector = SelectKBest(mutual_info_classif, k=2000)
    X_train_mi = mi_selector.fit_transform(X_train, y_train)
    X_test_mi = mi_selector.transform(X_test)

    tracemalloc.start()
    start_time = time.time()
    lr_mi_preds, lr_mi_report = run_logistic_regression(X_train_mi, X_test_mi, y_train, y_test)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results['MI+LR'] = {'accuracy': lr_mi_report['accuracy'], 'y_pred': lr_mi_preds}
    results_time_mem['MI+LR'] = {'time': elapsed, 'memory': peak / 1e6}
    print(f"MI+LR: {elapsed:.2f}s, {peak / 1e6:.2f} MB\n")

    # MI + SVD + LR
    print("MI + SVD + LR")
    svd_mi = TruncatedSVD(n_components=300, random_state=6942067)
    X_train_mi_svd = svd_mi.fit_transform(X_train_mi)
    X_test_mi_svd = svd_mi.transform(X_test_mi)

    tracemalloc.start()
    start_time = time.time()
    lr_mi_svd_preds, lr_mi_svd_report = run_logistic_regression(X_train_mi_svd, X_test_mi_svd, y_train, y_test)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results['MI+SVD+LR'] = {'accuracy': lr_mi_svd_report['accuracy'], 'y_pred': lr_mi_svd_preds}
    results_time_mem['MI+SVD+LR'] = {'time': elapsed, 'memory': peak / 1e6}
    print(f"MI+SVD+LR: {elapsed:.2f}s, {peak / 1e6:.2f} MB\n")

    # Plot results
    plot_accuracy_multi(results, "accuracy")
    plot_time_memory(results_time_mem)

