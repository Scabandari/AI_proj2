import sklearn
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.externals import joblib
import pickle


def random_forests(train_data,
                   train_labels,
                   test_data,
                   test_labels,
                   data_set1=True,
                   combined=None):

    N_ESTIMATORS = [500, 750, 1000]
    MAX_FEATURES = ["auto", "sqrt", "log2"]
    CRITERIION = ["gini", "entropy"]

    # N_ESTIMATORS = [500]
    # MAX_FEATURES = ["auto"]
    # CRITERIION = ["gini"]

    DECISION_TREE_ACCURACIES = {
        'Accuracy_train': 0,
        'Accuracy_test': 0
    }
    max_features_ = None
    n_estimators_ = None
    criterion_ = None

    for n_estimators, max_features, criterion in itertools.product(N_ESTIMATORS, MAX_FEATURES, CRITERIION):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            criterion=criterion)
        rf.fit(train_data, train_labels)

        pred_test = rf.predict(test_data)
        acc = accuracy_score(test_labels, pred_test)
        print("estimators: {} max_features: {} criterion: {} Accuracy: {}"
              .format(n_estimators, max_features, criterion, acc)
              )

        if acc > DECISION_TREE_ACCURACIES['Accuracy_test']:
            DECISION_TREE_ACCURACIES['Accuracy_test'] = acc # todo this line is new, test
            # DECISION_TREE_ACCURACIES['max_features'] = max_features
            # DECISION_TREE_ACCURACIES['n_estimators'] = n_estimators
            # DECISION_TREE_ACCURACIES['criterion'] = criterion
            max_features_ = max_features
            n_estimators_ = n_estimators
            criterion_ = criterion
            pred_train = rf.predict(train_data)
            acc_ = accuracy_score(train_labels, pred_train)
            DECISION_TREE_ACCURACIES['Accuracy_train'] = acc_
    #
    # rf = RandomForestClassifier(
    #     n_estimators=['n_estimators'],
    #     max_features=['max_features'],
    #     criterion=['criterion'])
    #rf = RandomForestClassifier()

    rf = RandomForestClassifier(
        n_estimators=n_estimators_,
        max_features=max_features_,
        criterion=criterion_)

    # todo 3 lines below are new, test and get entire code below working
    DECISION_TREE_ACCURACIES['max_features'] = max_features_
    DECISION_TREE_ACCURACIES['n_estimators'] = n_estimators_
    DECISION_TREE_ACCURACIES['criterion'] = criterion_

    if combined is not None:
        rf.fit(combined[0], combined[1])  # both first sets given, extra data == extra training
    else:
        rf.fit(train_data, train_labels)
    # save the trained model
    #file_name = 'ds1-randomForests.joblib' if data_set1 else 'ds2-randomForests.joblib'
    #joblib.dump(rf, file_name)  # todo make a pickle as well?

    file_name = 'ds1TEST-3.pkl' if data_set1 else 'ds2TEST-3.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(rf, file)

    return rf, DECISION_TREE_ACCURACIES
    #return rf  # , DECISION_TREE_ACCURACIES


# #######################################################################
#     n_estimators = None
#     max_features = None
#     if hyper_params:
#         try:
#             n_estimators = hyper_params['n_estimators']
#         except KeyError:
#             pass
#         try:
#             max_features = hyper_params['max_features']
#         except KeyError:
#             pass
#
#     if not n_estimators:
#         n_estimators = 100
#     if not max_features:
#         max_features = 200
#     rf = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_features=max_features)
#     rf.fit(train_data, train_labels)
#     score = rf.score(test_data, test_labels)
#     return {
#         'description': description,
#         'score': score
#     }
#
#
