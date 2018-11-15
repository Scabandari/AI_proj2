import numpy as np
import pandas as pd
import math
from random import randrange
import pickle


def combine_sets(sets):
    return pd.concat(sets)


def check_accuracy(model, test_features, test_labels):
    predictions = model.predict(test_features)
    predictions = predictions.tolist()
    correct = 0
    for index, data_point in enumerate(predictions):
        if test_labels[index] == data_point:
            correct += 1
    return correct/len(test_labels)


def model_fromPickle(file_name):
    model_ = None
    try:
        with open(file_name, 'rb') as modelfile:
            model_ = pickle.load(modelfile)
    except FileNotFoundError:
        pass
    if model_:
        return model_
    else:
        print("Could not load model from pickle")


def output_files(classifier_, test_set, file_name):
    """
    Run the classifier on the test set and create the output file with the results
    :param classifier:
    :param test_set:
    :param file_name: for output
    :return:
    """
    predictions = classifier_.predict(test_set)
    # print("predictions: {}".format(type(predictions)))
    # print(predictions)
    predictions = predictions.tolist()
    with open(file_name, 'w') as file:
        for index, num in enumerate(predictions):
            file.write("{},{}\n".format(index, num))


def add_data(train_set, percent_change, multiplier):
    """Take in the original data frame, copy the rows, change some values and add these new rows to data
    :param train_set: pandas data frame
    :param percent_change: how much to change each row. ie 0.05 for 5%,
    :param multiplier: the number of rows to add to the df for every original row
    :return: pandas dataframe of new augmented training data
    """
    # data = deepcopy(train_set)
    data = train_set.copy(deep=True)
    new_data = []
    for _ in range(multiplier):
        for index, row in data.iterrows():  # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
            new_row = row.copy(deep=True)
            # get the number of changes to make
            changes = int(math.floor(percent_change * new_row.size))
            for i in range(changes):
                index = randrange(0, new_row.size)
                new_row[index] = 1 if new_row[index] == 0 else 0
            new_data.append(new_row)
    new_data = np.array(new_data)
    new_data = pd.DataFrame(new_data, columns=train_set.columns)
    data = pd.concat([new_data, data])
    return data


def rename_cols(data_set):
    columns = []
    for i in range(len(data_set.columns)):
        columns.append(i)
    data_set.columns = columns


def separate_labels(train_set, test_set=None):
    """
    Take in training and test sets, remove last columnn as labels
    :param train_set:
    :param test_set:
    :return: train set,train labels, test set, test labels
    """
    yTrain = train_set[1024]  # our labels
    xTrain = train_set.iloc[:, :-1]  # our feature cols

    if test_set is not None:
        yTest = test_set[1024]
        xTest = test_set.iloc[:, :-1]
        return xTrain, yTrain, xTest, yTest

    return xTrain, yTrain


# I don't think we'll need this but not sure yet
def variance_threshold(train_set, test_set, threshold_val, solution_set=None):
    """
     This function takes training set and testing set, combines them then
        drops unnecessary columns, separates them back into train & test and
        returns them
    :param train_set:
    :param test_set:
    :param threshold_val:
    :param solution_set: If we need this for demo on solution set will have to update code below
            to drop cols on sol set also
    :return: new training and testing sets
    """
    train_rows, train_cols = train_set.shape
    # print("train_set.shape {}".format(train_set.shape))
    # print("test_set.shape {}".format(test_set.shape))

    # if solution_set:  # todo if this works need to finish implement if solution_set != None
    #     sol_set_rows, sol_set_cols = solution_set.shape
    #     print("solution_set.shape: {}".format(solution_set.shape))
    # if solution_set:
    #     combined_set = pd.concat([train_set, test_set, solution_set])

    combined_set = pd.concat([train_set, test_set])
    thresholder = VarianceThreshold(threshold=(threshold_val * (1 - threshold_val)))
    thresholder.fit(combined_set)
    col_names = thresholder.get_support(indices=True)
    # transorming the dataframe changes from pandas df to n-dimensional numpy array
    # change it back
    transformed_df = thresholder.transform(combined_set)
    transformed_df = pd.DataFrame(transformed_df, columns=col_names)
    transformed_train = transformed_df[0: train_rows]
    #print("type transformed_train: {}".format(type(transformed_train)))
    transformed_test = transformed_df[train_rows: ]

    # for some reason these sets below aren't the same, would be a problem unless we can include the final
    # solution set before we transform then separate after as done above
    # check_train = train_set[col_names]
    # print("check_train.head: {}".format(check_train.head()))
    # print("transformed_train.head: {}".format(transformed_train.head()))
    #
    # print("transformed_train.shape: {}".format(transformed_train.shape))
    # print("transformed_test.shape: {}".format(transformed_test.shape))

    return transformed_train, transformed_test


# def see_answers(features):
#     cols = features.columns
#     for index, row in features.iterrows():
#         for i, col in enumerate(cols):
#             print(row[index], end="")
#             if i % 32 == 0:
#                 print("\n")
#         print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
