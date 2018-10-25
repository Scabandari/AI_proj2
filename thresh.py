import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# IGNORE THIS FILE FOR NOW, JUST DUMPED SOME OLD CODE

"""Variance thresholding may remove some columns of our data if all values in that column are
    overly similar ie don't vary:
     https://chrisalbon.com/machine_learning/feature_selection/variance_thresholding_binary_features/"""

"""See here: https://www.programcreek.com/python/example/93973/sklearn.feature_selection.VarianceThreshold
    Ex1. get support about keeping column names so we can know which columns to drop in the test data"""

# todo refactor this file to export 1 function that takes data files as input and outputs





# todo work the ability to give multiple outputs of above function based on multiple
# todo inputs for threshold_levels?
# todo maybe at some point but not pressing at the moment

# CODE DUMP BELOW, ignore this
# thresh_train, thresh_test = variance_threshold(data_train, data_test, 0.75)
# threshold_levels = [0.25, 0.5, 0.75, 0.85, 0.9]
# th_experiments = []
# for level in threshold_levels:
#     thresh_train, thresh_test = variance_threshold(data_train, data_test, level)
#     experiment = {
#         'level': level,
#         'train': thresh_train,
#         'test': thresh_test
#     }
#     th_experiments.append(experiment)
#     # todo unfinished?

# set1_yTrain_thresh = thresh_train[1024]
# # print("set1_yTrain_thresh: ")
# # print(set1_yTrain_thresh)
# set1_xTrain_thresh = thresh_train.iloc[:, :-1]
# # print("thresh_train.head(): ")
# # print(thresh_train.head())
#
#
# set1_yTest_thresh = thresh_test[1024]  # our labels
# set1_xTest_thresh = thresh_test.iloc[:, :-1]  # our features
# print("set1_xTest_thresh.head(): ")
# print(set1_xTest_thresh.head())






