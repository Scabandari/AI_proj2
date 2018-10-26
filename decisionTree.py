from sklearn import tree
from sklearn.metrics import accuracy_score


def decision_tree(samples, features, test_samples, test_features, description):

    clf = tree.DecisionTreeClassifier()
    clf.fit(samples, features)

    prediction = clf.predict(test_samples)

    accuracy = accuracy_score(test_features, prediction)

    return {
        'description': description,
        'score': accuracy
    }


