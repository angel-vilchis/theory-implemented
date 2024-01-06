from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs, make_regression
from helpers.MathFunctions import logistic

def classification_split(n_samples=300, n_features=4, n_classes=2, n_clusters_per_class=1, random_state=42, test_size=0.2, binary_features=False):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, random_state=random_state)
    if binary_features:
        X = (logistic(X) > 0.5).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def clustering_split(n_samples=300, n_features=4, centers=3, cluster_std=1.5, random_state=42, test_size=0.2):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def regression_split(n_samples=300, n_features=4, noise=20, random_state=42, test_size=0.2):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test