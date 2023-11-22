# SUMO-NN : a neural networks based method to predict SUMOylated lysines in human proteins

This method takes protein sequence as input and predicts lysine residues that are likely to undergo SUMOylation. SUMO-NN is implemented in Python version 3.11.6 and PyTorch version 2.0.1+cpu. More details related to the method can be found in the file named "Project_Report.pdf".

The folder named "dataset_creation" contains programs and data files related to data curation and generation of training and testing datasets. Training and testing datasets can be found in files named "positive_training_data.tsv", "negative_training_data.tsv", "positive_testing_data.tsv" and "negative_testing_data.tsv" respectively.

The folder named "src" contains the program "predict_lysines.py". This is the most important program of this project and it contains all the necessary hyperparameters and architecture of SUMO-NN. To run this program, run the following command - 

python3 predict_lysines.py

This program trains the neural network and makes predictions on test data. Predictions are saved to a file named "predictions_on_test_data.tsv".

The folder named "performance_assessment" contains the program "get_performance.py". This program does statistical analysis of the quality of predictions made by SUMO-NN.
