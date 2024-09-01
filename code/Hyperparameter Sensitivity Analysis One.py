from utility import evaluate_model, print_evaluation_results
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
dataset = 'heart_disease'
from utility import load_data
data, target = load_data(dataset)


estimators = [10, 20, 30, 40, 50]

for estimator in estimators:
    # Random Forest
    extra_trees_results = evaluate_model(RandomForestClassifier(random_state=23, n_estimators=estimator), data, target, k_folds=10)
    print_evaluation_results(extra_trees_results)

    # Extra Trees
    extra_trees = ExtraTreesClassifier(random_state=23, n_estimators=estimator)
    extra_trees_results = evaluate_model(extra_trees, data, target, k_folds=10)
    print_evaluation_results(extra_trees_results)

    #Totally Random Forest
    trf = RandomForestClassifier(random_state=42, bootstrap=False, max_features=None, n_estimators=estimator)
    trf_results = evaluate_model(trf, data, target, k_folds=10)
    print_evaluation_results(trf_results)

    # Adaboost Random Forest
    adrf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=estimator, random_state=42))
    adrf_results = evaluate_model(adrf, data, target, k_folds=10)
    print_evaluation_results(adrf_results)

max_depths = [1, 2, 3, 4, 5, None]

for depth in max_depths:
    # Random Forest
    extra_trees_results = evaluate_model(RandomForestClassifier(random_state=23, max_depth=depth), data, target, k_folds=10)
    print_evaluation_results(extra_trees_results)

    # Extra Trees
    extra_trees = ExtraTreesClassifier(random_state=23, max_depth=depth)
    extra_trees_results = evaluate_model(extra_trees, data, target, k_folds=10)
    print_evaluation_results(extra_trees_results)

    #Totally Random Forest
    trf = RandomForestClassifier(random_state=42, bootstrap=False, max_features=None, max_depth=depth)
    trf_results = evaluate_model(trf, data, target, k_folds=10)
    print_evaluation_results(trf_results)

    # Adaboost Random Forest
    adrf = AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=depth, random_state=42))
    adrf_results = evaluate_model(adrf, data, target, k_folds=10)
    print_evaluation_results(adrf_results)