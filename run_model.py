from decisionTree import predict

# test set file in csv format
testset_filename = "ds1Test.csv"
# classifier model in pickle format
model_filename = "DTmodel1.pkl"
# name of prediction result file to be created
test_result_filename = "result.csv"

predict(testset_filename, model_filename, test_result_filename)
