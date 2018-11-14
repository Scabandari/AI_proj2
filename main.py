import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from copy import deepcopy
from utils import add_data, separate_labels, rename_cols
from naiveBayes import bernoulli

""" 
We're having issues that I'm sure everyone is, our predictions are not accurate. 
K-folds should help. We're allowed to use it? Our train and test data is seperate,
with k-folds normally you treat it as one big group and the sectioning gets 
automated in the algo
"""

# list of dicts for each classifier {'description': "blah blah", 'score': score}
# description should describe hyper params as well
results = {'naive bayes': [],
           'trees': [],
           'random forests': []}


def print_results():
    for key in results.keys():
        for experiment_ in results[key]:
            print("{}: {}".format(
                experiment_['description'],
                experiment_['score'])
            )


# so we can rename the columns of this df, no idea what the previous column names mean
columns = []
for i in range(1025):
    columns.append(i)

data_train = pd.read_csv('ds1Train.csv', delimiter=',')
rename_cols(data_train)
data_test = pd.read_csv('ds1Val.csv', delimiter=',')
rename_cols(data_test)

set1_xTrain, set1_yTrain, set1_xTest, set1_yTest = separate_labels(data_train, data_test)

# # Naive Bayes
results['naive bayes'].append(bernoulli(
    set1_xTrain,
    set1_yTrain,
    set1_xTest,
    set1_yTest,
    "Naive Bayes Bernoulli"))

# now we can do basically the same thing again after we've augmented our training data
# by copying all the rows making small changes in the features but keeping the labels
# since the small changes shouldn't affect which letter the instance is most like
# AUGMENTING DATA NOT HELPING SO FAR
# augmented_train = add_data(data_train, 0.08, 20)
# rename_cols(augmented_train)
# aug_xTrain, aug_yTrain = separate_labels(augmented_train)
# results['naive bayes'].append(bernoulli(
#     aug_xTrain,
#     aug_yTrain,
#     set1_xTest,
#     set1_yTest,
#     "Naive Bayes Bernoulli with augmented training data"))

print_results()

"""Results not very good. Try Random forests"""

"""Decision trees """
accuracy = {}
max_depths = []
test_accuracies = []

# for i in range(6, 40, 2):
#     max_depths.append(i)
#     tree = DecisionTreeClassifier(max_depth=i, random_state=0)
#     tree.fit(set1_xTrain, set1_yTrain)
#     train = tree.score(set1_xTrain, set1_yTrain)
#     test = tree.score(set1_xTest, set1_yTest)
#     test_accuracies.append(test)
#     data = {'max depth': i,
#      'train': train,
#      'test': test}
#     accuracy[i] = data

# for data in accuracy.keys():
#     print("\n\nMax depth: {}".format(accuracy[data]['max depth']))
#     print("Accuracy of Decision Tree classifier on training set is: {:.3f}"
#           .format(accuracy[data]['train']))
#     # above accuracy is .99 which mean overfitting
#     print("Accuracy of Decision Tree classifier on training setis: : {:.3f}"
#           .format(accuracy[data]['test']))
#
# plt.barh(range(len(max_depths)), test_accuracies, align='center')
# plt.yticks(np.arange(len(max_depths)), max_depths)
# plt.xlabel("Accuracy")
# plt.ylabel("Max depth")
# plt.show()





"""Random forests are basically a collection of decision trees, parameter ?max_forests?
    can be changed to control the randomness of each tree"""
# forest = RandomForestClassifier(n_estimators=100, random_state=0)
# forest.fit(set1_xTrain, set1_yTrain)
# print("Accuracy score for Random forests on training data set 1: {}"
#       .format(forest.score(set1_xTrain, set1_yTrain)))
# # print("Accuracy score for Random forests on testing data set 1: {}".format(forest.score(set1_xTest, set1_yTest)))
