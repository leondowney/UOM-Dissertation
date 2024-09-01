import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")


def load_data(name, normalize=True, test_size=0.2, random_state=42):
    if name == 'cancer':
        cancer = datasets.load_breast_cancer()
        data = cancer.data
        target = cancer.target
    elif name == 'diabetes':
        diabetes = pd.read_csv('data/diabetes.csv')
        data = diabetes.drop('Outcome', axis=1)
        target = diabetes['Outcome']
    elif name == 'heart_disease':
        columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                   "ca", "thal", "target"]
        heart_disease = pd.read_csv('data/heart-disease.data', names=columns)
        heart_disease = heart_disease.replace('?', np.nan)
        heart_disease = heart_disease.dropna()
        data = heart_disease.drop('target', axis=1)
        target = heart_disease['target']
    else:
        raise ValueError('Invalid dataset name')

    if normalize:
        data = StandardScaler().fit_transform(data)

    return data, target


def evaluate_model(model, X, y, k_folds=5, n_jobs=-1):
    def single_run(train_index, test_index, model, X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        return accuracy, precision, recall, f1

    X = np.array(X)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=k_folds)
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_run)(train_index, test_index, model, X, y) for train_index, test_index in skf.split(X, y))

    accuracies, precisions, recalls, f1s = zip(*results)

    mean_accuracies = np.mean(accuracies)
    mean_precisions = np.mean(precisions)
    mean_recalls = np.mean(recalls)
    mean_f1s = np.mean(f1s)

    std_accuracies = np.std(accuracies)
    std_precisions = np.std(precisions)
    std_recalls = np.std(recalls)
    std_f1s = np.std(f1s)

    return {
        "accuracy": (mean_accuracies, std_accuracies),
        "precision": (mean_precisions, std_precisions),
        "recall": (mean_recalls, std_recalls),
        "f1": (mean_f1s, std_f1s)
    }


def print_evaluation_results(results, decimal_places=2):
    for metric, (mean, std) in results.items():
        rounded_mean = round(mean * 100, decimal_places)
        rounded_std = round(std * 100, decimal_places)
        print(f"{metric.capitalize()}: {rounded_mean:.{decimal_places}f} \pm {rounded_std:.{decimal_places}f}")
