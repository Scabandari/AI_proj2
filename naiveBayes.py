import pandas as pd
import numpy as np
import itertools
from sklearn.externals import joblib
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import pickle

"""Naive bayes: https://www.youtube.com/watch?v=99MN-rl8jGY
    bernoulie ==> binary features but we can try the other 2 kinds for 
    experimentation"""


def bernoulli(train_data, train_labels, test_data, test_labels, data_set1=True, combined=None):
    """
    Performs the Naive Bayes Bernoulli classification on the test data after training the model
    on the training data
    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :param description:
    :return:
    """

    DECISION_TREE_ACCURACIES = {
        'Accuracy_train': 0,
        'Accuracy_test': 0
    }
    ALPHA = [0, .01, .025, .05, .075, 0.1, 0.2, 0.3, .5, .75, 1, 1.5, 2.5]
    #ALPHA = [0, 0.175, 0.190, 0.195, 0.2, 0.205, 0.21, 0.225]

    FIT_PRIOR = [True, False]

    for alpha, fit_prior in itertools.product(ALPHA, FIT_PRIOR):
        bern = BernoulliNB(alpha=alpha, fit_prior=fit_prior)
        bern.fit(train_data, train_labels)

        pred_test = bern.predict(test_data)
        acc = accuracy_score(test_labels, pred_test)
        print("Alpha: {} Fit Prior: {} Accuracy: {}".format(alpha, fit_prior, acc))

        if acc > DECISION_TREE_ACCURACIES['Accuracy_test']:
            DECISION_TREE_ACCURACIES['Accuracy_test'] = acc  # todo this line is new, test
            DECISION_TREE_ACCURACIES['Alpha'] = alpha
            DECISION_TREE_ACCURACIES['Fit_prior'] = fit_prior
            pred_train = bern.predict(train_data)
            acc_ = accuracy_score(train_labels, pred_train)
            DECISION_TREE_ACCURACIES['Accuracy_train'] = acc_

    bern = BernoulliNB(alpha=DECISION_TREE_ACCURACIES['Alpha'],
                       fit_prior=DECISION_TREE_ACCURACIES['Fit_prior'])

    if combined is not None:
        bern.fit(combined[0], combined[1])  # both first sets given, extra data == extra training
    else:
        bern.fit(train_data, train_labels)

    # save the trained model
    file_name = 'ds1TEST-nb.pkl' if data_set1 else 'ds2TEST-nb.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(bern, file)

    return bern, DECISION_TREE_ACCURACIES



