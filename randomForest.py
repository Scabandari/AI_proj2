import pickle

from sklearn.ensemble import RandomForestClassifier
from decisionTree import load_csv, predict
from sklearn.metrics import accuracy_score
import numpy as np
import itertools


def random_forest(features, labels, test_features, test_labels, model_save_filename):

    DECISION_TREE_ACCURACIES = {
        'Accuracy': 0
    }
    N_ESTIMATOR = list(np.arange(10, 30, 5))
    MAX_FEATURES = list(np.arange(0.2, 0.6, 0.1))
    MIN_SAMPLES_LEAF = list(np.arange(1, 5))

    for n_estimators, max_features, min_samples_leaf in itertools.product(N_ESTIMATOR, MAX_FEATURES, MIN_SAMPLES_LEAF):
        classifier = RandomForestClassifier(max_features=max_features,
                                            n_estimators=n_estimators,
                                            min_samples_leaf=min_samples_leaf,
                                            random_state=0)
        classifier.fit(features, labels)
        prediction = classifier.predict(test_features)
        accuracy = accuracy_score(test_labels, prediction)
        # print(accuracy)
        if accuracy > DECISION_TREE_ACCURACIES['Accuracy']:
            DECISION_TREE_ACCURACIES['Accuracy'] = accuracy
            DECISION_TREE_ACCURACIES['Max_features'] = max_features
            DECISION_TREE_ACCURACIES['Min_samples_leaf'] = min_samples_leaf
            DECISION_TREE_ACCURACIES['N_estimator'] = n_estimators

    classifier = RandomForestClassifier(max_features=DECISION_TREE_ACCURACIES['Max_features'],
                                        n_estimators=DECISION_TREE_ACCURACIES['N_estimator'],
                                        min_samples_leaf=DECISION_TREE_ACCURACIES['Min_samples_leaf'],
                                        random_state=0)
    classifier.fit(features, labels)
    with open(model_save_filename, 'wb') as file:
        pickle.dump(classifier, file)

    return DECISION_TREE_ACCURACIES



if __name__ == "__main__":
    features, labels, test_features, test_labels = load_csv('ds2Train.csv', 'ds2Val.csv')
    # rf_parameters = random_forest(features, labels, test_features, test_labels, 'RFmodel1.pkl')
    # print(rf_parameters)
    # predict('ds1Test.csv', 'RFmodel1.pkl', 'ds1Test-3.csv')

    for i in list(np.arange(0.2, 0.8, 0.1)):
        classifier = RandomForestClassifier(random_state=0, max_features=i)
        classifier.fit(features, labels)
        prediction = classifier.predict(test_features)
        accuracy = accuracy_score(test_labels, prediction)
        print(i, accuracy)
