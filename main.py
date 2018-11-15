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
from naiveBayes import bernoulli
from randomForests import random_forests
#from sklearn.externals import joblib
import pickle
from copy import deepcopy

from utils import (add_data,
                   separate_labels,
                   rename_cols,
                   output_files,
                   model_fromPickle,
                   check_accuracy,
                   combine_sets)


columns = []
for i in range(1025):
    columns.append(i)

# read in data set 1 Train and Val and Test, rename the columns
data_train = pd.read_csv('ds1Train.csv', delimiter=',')
rename_cols(data_train)
data_test = pd.read_csv('ds1Val.csv', delimiter=',')
rename_cols(data_test)
ds1Test = pd.read_csv('ds1Test.csv', delimiter=',')
rename_cols(ds1Test)


# Seperate into features + labels for Data set 1
set1_xTrain, set1_yTrain, set1_xTest, set1_yTest = separate_labels(data_train, data_test)

feature_sets = [set1_xTrain, set1_xTest]
label_sets = [set1_yTrain, set1_yTest]

set1_features = combine_sets(feature_sets)
set1_labels = combine_sets(label_sets)
combined_ds1 = [set1_features, set1_labels]

# load previously classified models

files_out = [
    # 'ds1TEST-dt',
    # 'ds2TEST-dt',
    'ds1TEST-nb',
    # 'ds2TEST-nb',
    'ds1TEST-3' #,
    #'d2TEST-3'
]

trained_models = []   # testing only

for file_name in files_out:
    f = deepcopy(file_name)
    f += ".pkl"
    model = model_fromPickle(f)
    if model is not None:
        trained_models.append(model)  # testing only
        f = deepcopy(file_name)
        f += ".csv"
        output_files(model, ds1Test, f)

########## testing only #####################################
for index, model in enumerate(trained_models):
    accuracy = check_accuracy(model, set1_xTest, set1_yTest)
    print("Model {}: {}".format(index, accuracy))
##############################################################


# create dict to store some meta-data on params and performance after tuning in case we want access
# later for summary statistics
best = {
    'naive bayes': {},
    'decision trees': {},
    'random forests': {}
}

# Get classifier and meta data on Naive Bayes for ds1
bern, best['naive bayes'] = bernoulli(set1_xTrain,
                                      set1_yTrain,
                                      set1_xTest,
                                      set1_yTest,
                                      data_set1=True,
                                      combined=combined_ds1)
output_files(bern, set1_xTrain, 'ds1Test-nb.csv')  #todo should be final test set here (set1_features)

# Get classifier and meta data on Random Forests
#rf = random_forests(set1_xTrain, set1_yTrain, set1_xTest, set1_yTest, data_set1=True)
rf, best['random forests'] = random_forests(set1_xTrain,
                                            set1_yTrain,
                                            set1_xTest,
                                            set1_yTest,
                                            data_set1=True,
                                            combined=combined_ds1)
output_files(rf, set1_xTrain, 'ds1Test-3.csv')  #todo should be final test set here


####################################################################################
