import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.utils import Bunch
from DataframeToSklearn import load_pima, load_heart
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

LABEL_IDX = 0
PRED_IDX = 1

# Loading the dataset
breast_data = load_breast_cancer()

def compareRepeat(baseline_result, pred_result) -> np.ndarray:
    def single_run(pred_idx, baseline_result, pred_result):
        different_count = np.sum(baseline_result != pred_result[pred_idx])
        return different_count

    res_list = Parallel(n_jobs=-1)(
        delayed(single_run)(pred_idx=i, baseline_result=baseline_result, pred_result=pred_result) for i in
        range(len(pred_result)))
    return np.vstack(res_list)


def evaluate_model(model, X, y, k_folds=5, n_jobs=-1):
    def single_run(train_index, test_index, model, X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    X = np.array(X)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=k_folds)
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_run)(train_index, test_index, model, X, y) for train_index, test_index in skf.split(X, y))
    max_length = max(len(arr) for arr in results)
    padded_results = [np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in results]

    return np.vstack(padded_results)


# load dataset
data = load_breast_cancer()

svm = SVC(kernel='rbf')
svm_results = evaluate_model(svm, data.data, data.target, k_folds=5, n_jobs=-1)[0]

eva_list = []

# max_depth
max_depths = [1, 2, 3, 4, 5, None]
for i in max_depths:

    #ADRF
    classifier = AdaBoostClassifier(estimator=RandomForestClassifier(max_depth=i, random_state=23),
     n_estimators=5, random_state=42)

    results = evaluate_model(classifier, data.data, data.target, k_folds=5, n_jobs=-1)
    different_res = compareRepeat(svm_results, results)
    eva_list.append(different_res.mean() / len(data.target) * 100)


# drawing
plt.figure(figsize=(6, 6))
colors = sns.color_palette("viridis", len(eva_list))

plt.bar(x=[i for i in range(1, len(eva_list) + 1)], height=eva_list, width=1, color=colors)

x = np.array([i for i in range(1, len(eva_list) + 1)])
y = np.array(eva_list)
# create uni-variate spline
spl = UnivariateSpline(x, y)


xnew = np.linspace(x.min(), x.max(), 500)

ynew = spl(xnew)

plt.plot(xnew, ynew, color='blue')

plt.scatter(x, y, color='blue')

fmt = '%.2f%%'
yticks = mtick.FormatStrFormatter(fmt)
plt.gca().yaxis.set_major_formatter(yticks)
plt.ylim(bottom=3.8)

for i, v in enumerate(eva_list):
    plt.text(i + 1, v + 0.03, f"{v:.3f}%", ha='center', va='top')

# Setting the title and axis labels
plt.title('Different Sample Percentage(ADRF)', fontsize=16)
plt.xlabel('Max Depth', fontsize=14)
plt.gca().set_xticklabels(['0','1', '2', '3', '4', '5', 'None'])

# Increase tick font size
plt.tick_params(labelsize=12)
plt.savefig('ADRF_Max_Depth.png')

# Display Graphics
plt.show()
