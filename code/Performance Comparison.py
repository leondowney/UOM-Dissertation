from utility import evaluate_model, print_evaluation_results
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
# ## Load Data
# datasets: cancer, diabetes, heart_disease
dataset = 'heart_disease'
from utility import load_data

data, target = load_data(dataset)

# MLP
mlp = MLPClassifier(random_state=42, max_iter=800)
mlp_results = evaluate_model(mlp, data, target, k_folds=10)
print_evaluation_results(mlp_results)

# Bootstrap Aggregating

bagging = BaggingClassifier(random_state=42)
bagging_results = evaluate_model(bagging, data, target, k_folds=10)
print_evaluation_results(bagging_results)

# SVM
svm = SVC(kernel='rbf')
svm_results = evaluate_model(svm, data, target, k_folds=10)
print_evaluation_results(svm_results)



# Random Forest
extra_trees_results = evaluate_model(RandomForestClassifier(random_state=23), data, target, k_folds=10)
print_evaluation_results(extra_trees_results)

# Extra Trees
extra_trees = ExtraTreesClassifier(random_state=23)
extra_trees_results = evaluate_model(extra_trees, data, target, k_folds=10)
print_evaluation_results(extra_trees_results)

#Totally Random Forest
trf = RandomForestClassifier(random_state=42, bootstrap=False, max_features=None)
trf_results = evaluate_model(trf, data, target, k_folds=10)
print_evaluation_results(trf_results)

# Adaboost Random Forest
adrf = AdaBoostClassifier(base_estimator=RandomForestClassifier(), n_estimators=100, random_state=42)
adrf_results = evaluate_model(adrf, data, target, k_folds=10)
print_evaluation_results(adrf_results)

