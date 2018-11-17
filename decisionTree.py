from sklearn import tree
from sklearn.metrics import accuracy_score
import csv
import numpy as np
import itertools
import pickle


def load_csv(train_data_filename, validation_data_filename):
    pass
    with open(train_data_filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        data = []
        for row in reader:
            # print(row)
            new_row = [int(number) for number in row]
            data.append(new_row)
        file.close()
        train_features = [d[:-1] for d in data]
        train_labels = [d[-1] for d in data]

    with open(validation_data_filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        data = []
        for row in reader:
            new_row = [int(number) for number in row]
            data.append(new_row)
    file.close()

    test_features = [d[:-1] for d in data]
    test_labels = [d[-1] for d in data]

    return train_features, train_labels, test_features, test_labels


def decision_tree(features, labels, test_features, test_labels, model_save_filename):
    DECISION_TREE_ACCURACIES = {
        'Accuracy': 0
    }

    # source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    CRITERION = ['gini', 'entropy']
    SPLITTER = ['best', 'random']
    MAX_DEPTH = list(np.arange(10, 60, 5))
    MAX_DEPTH.append(None)
    # MIN_SAMPLES_SPLIT.append(2)
    MIN_SAMPLES_LEAF = list(np.arange(1, 5))
    MAX_FEATURES = list(np.arange(0.1, 0.5, 0.1))

    for criterion, splitter, max_depth, min_samples_leaf, max_features in itertools.product(
            CRITERION, SPLITTER, MAX_DEPTH, MIN_SAMPLES_LEAF, MAX_FEATURES):
        classifier = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                 min_samples_leaf=min_samples_leaf,
                                                 max_features=max_features,
                                                 random_state=0)
        # print(max_features)
        classifier.fit(features, labels)


        validation_predicted = classifier.predict(test_features)
        # print('finished prediction')
        # print(validation_predicted)
        accuracy = accuracy_score(test_labels, validation_predicted)
        # print(accuracy)

        if accuracy > DECISION_TREE_ACCURACIES['Accuracy']:
            DECISION_TREE_ACCURACIES['Accuracy'] = accuracy
            DECISION_TREE_ACCURACIES['Criterion'] = criterion
            DECISION_TREE_ACCURACIES['Splitter'] = splitter
            DECISION_TREE_ACCURACIES['Max_depth'] = max_depth
            DECISION_TREE_ACCURACIES['Min_sample_leaf'] = min_samples_leaf
            DECISION_TREE_ACCURACIES['Max_features'] = max_features

    classifier = tree.DecisionTreeClassifier(criterion=DECISION_TREE_ACCURACIES['Criterion'],
                                             splitter=DECISION_TREE_ACCURACIES['Splitter'],
                                             max_depth=DECISION_TREE_ACCURACIES['Max_depth'],
                                             min_samples_leaf=DECISION_TREE_ACCURACIES['Min_sample_leaf'],
                                             max_features=DECISION_TREE_ACCURACIES['Max_features'],
                                             random_state=0)
    classifier.fit(features, labels)
    with open(model_save_filename, 'wb') as file:
        pickle.dump(classifier, file)

    return DECISION_TREE_ACCURACIES


def predict(testset_filename, model_filename, test_result_filename):
    pass
    # load testset features
    with open(testset_filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        testset_data = []
        for row in reader:
            # print(row)
            new_row = [int(number) for number in row]
            testset_data.append(new_row)
    file.close()

    # load model
    with open(model_filename, 'rb') as modelfile:
        classifier = pickle.load(modelfile)
    modelfile.close()

    prediction = classifier.predict(testset_data)
    with open(test_result_filename, 'w') as result_file:
        for i in range(len(prediction)):
            result_file.write('%d,%d\n' % (i + 1, prediction[i]))


if __name__ == "__main__":
    features, labels, test_features, test_labels = load_csv('ds1Train.csv', 'ds1Val.csv')  # load training data and validation data
    dt_parameters = decision_tree(features, labels, test_features, test_labels, 'DTmodel1.pkl')  # find best tuned model and save to filename
    print(dt_parameters)
    predict('ds1Test.csv', 'DTmodel1.pkl', 'ds1Test-dt.csv')  # run prediction on testset, using saved model, and save test result to file

    # for i in range(0, 110, 10):
    #     classifier = tree.DecisionTreeClassifier(random_state=i)
    #     classifier.fit(features, labels)
    #     prediction = classifier.predict(test_features)
    #     print(i, accuracy_score(test_labels, prediction))

    # with open('DTmodel2.pkl', 'rb') as modelfile:
    #     classifier = pickle.load(modelfile)
    # modelfile.close()
    # prediction = classifier.predict(test_features)
    # with open('ds2Val-dt.csv', 'w') as result_file:
    #     for i in range(len(prediction)):
    #         result_file.write('%d,%d\n' % (i + 1, prediction[i]))
    # result_file.close()
