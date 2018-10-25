import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

"""Naive bayes: https://www.youtube.com/watch?v=99MN-rl8jGY
    bernoulie ==> binary features but we can try the other 2 kinds for 
    experimentation"""


def bernoulli(train_data, train_labels, test_data, test_labels, description):
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
    bern = BernoulliNB()
    bern.fit(train_data, train_labels)
    # ignore below for now, can find out if model is overfit by testing on training set?
    # y_pred_train = bern.predict(set1_xTrain)
    # acc = accuracy_score(set1_yTrain, y_pred_train)
    # print("Accuracy score for bernouli naive bayes on training data set 1: {}".format(acc))

    # now for the test set, should see lower accuracy
    pred_test = bern.predict(test_data)
    acc = accuracy_score(test_labels, pred_test)
    return {
        'description': description,
        'score': acc
    }