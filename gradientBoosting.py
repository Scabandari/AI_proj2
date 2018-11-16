import sklearn
from sklearn.ensemble import GradientBoostingClassifier


def gradient_boosting(train_data,
                      train_labels,
                      test_data,
                      test_labels,
                      description,
                      hyper_params=None):

    # n_estimators = None
    # max_features = None
    # if hyper_params:
    #     try:
    #         n_estimators = hyper_params['n_estimators']
    #     except KeyError:
    #         pass
    #     try:
    #         max_features = hyper_params['max_features']
    #     except KeyError:
    #         pass
    #
    # if not n_estimators:
    #     n_estimators = 100
    # if not max_features:
    #     max_features = 200
    #
    gb = GradientBoostingClassifier()
    gb.fit(train_data, train_labels)
    score = gb.score(test_data, test_labels)
    return {
        'description': description,
        'score': score
    }
