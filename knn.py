# THIS FILE IS JUST A CODE DUMP, DIDN'T WANT TO ERASE TOTALLY, LEAVE FOR NOW
# todo all this code was in main.py but now moving most code to other files
# todo will pass data and results dict to functions pulled from other files
# todo to keep things organized

# todo turn this code into a function but DO WE EVEN NEED THIS?
# todo preliminary results would suggest knn not good enough to be our 3rd classifier
# todo leave this for now


number_neighbors = [5, 10, 15, 20]

for num_ in number_neighbors:
    knn = KNeighborsClassifier(n_neighbors=10)  # n_neighbors would be the parameter to be tuned                                                # not sure if there are others
    knn.fit(set1_xTrain, set1_yTrain)
    score = knn.score(set1_xTest, set1_yTest)
    desc = "KNN with number of neighbors: {}".format(num_)
    exp_ = {
        'description': desc,
        'score': score
    }
    results['knn'].append(exp_)

print_results()  # from main.py, watch the circular imports

"""KNN k defines the number of neighboring points(data row) to examine in order to make an estimate
    of which one our new data point is the most like. Default is 5"""

# print("\nAccurancy of KNN n=5 on training set: {:.3f}".format(knn.score(set1_xTrain, set1_yTrain)))
# print("\nAccurancy of KNN n=5 on test set: {:.3f}".format(knn.score(set1_xTest, set1_yTest)))


# knn_th = KNeighborsClassifier(n_neighbors=10)
# knn_th.fit(set1_xTrain_thresh, set1_yTrain_thresh)
# print("\nAccurancy: KNN n=5 , threshold=0.75 on training set after thresholding: {:.3f}"
#       .format(knn_th.score(set1_xTrain_thresh, set1_yTrain_thresh)))
# print("\nAccurancy: KNN n=5 , threshold=0.75 on testing set after thresholding:{:.3f}"
#       .format(knn_th.score(set1_xTest_thresh, set1_yTest_thresh)))

