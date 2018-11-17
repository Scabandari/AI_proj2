**Before using, please installed the required libs:**
	pip install -r requirements.txt

	
TO RUN PREDICTIONS:

- Edit the script 'run_model.py', change the values of testset_filename, model_filename, and test_result_filename
- Make sure testset_filename and model_filename point to existing files
- Compile and run
- Please find test result file in the same directory as the script

TO REPRODUCE THE EXPERIMENTS:

Under Linux cd into the project directory. We have 2 files that need to be 
run in order to reproduce our results.

Running main.py will produce the results for Naives Bayes and our 
3rd option Random Forests.

Running decisionTree.py will produce the results for Decision Tree.

The project is written in Python3 so while in project directory run:

python3 main.py

Examine results and then run:

python3 decisionTree.py 

Manual tweakings are necessary to run experiment on different datasets. For example:
- to run different Datasets using Decision Tree, open decisionTree.py
- modify the file names in the bottom __main__ section of the file
- save, compile, and execute