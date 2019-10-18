import numpy as np
from l2_distance import l2_distance
import utils
from plot_digits import *
import matplotlib 

def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:,:k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels

if __name__ == '__main__':
    train_inputs, train_targets = utils.load_train()
    valid_inputs, valid_targets = utils.load_valid()
    test_inputs, test_targets = utils.load_test()
    
    set_k = [1,3,5,7,9]
    
    accuracy_valid_output = {}
    accuracy_test_output = {}
   
    length_valid = len(valid_inputs)
    length_test = len(test_inputs)
    
    for k in set_k:
        valid_outputs = run_knn(k, train_inputs, train_targets, valid_inputs)
        test_outputs =  run_knn(k, train_inputs, train_targets, test_inputs)

        count_valid = np.sum(valid_outputs == valid_targets) 
        accuracy_valid = count_valid/length_valid
        accuracy_valid_output["k="+str(k)] = accuracy_valid
        
        count_test = np.sum(test_outputs == test_targets) 
        accuracy_test = count_test/length_test
        accuracy_test_output["k="+str(k)] = accuracy_test
        
    print("Validation Accuracy:")
    print(accuracy_valid_output)
    print("\n")
    print("Test Accuracy: ")
    print(accuracy_test_output)

    matplotlib.pyplot.plot(set_k, list(accuracy_valid_output.values()), label="validation rate")
    matplotlib.pyplot.plot(set_k, list(accuracy_test_output.values()), label="test rate")
    
    matplotlib.pyplot.title("Test VS Validation Classification Rate")
    matplotlib.pyplot.xlabel("k")
    matplotlib.pyplot.ylabel("Accuracy/Classification Rate")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig("Q2_2 Test Set VS Validation Rates.png")
    matplotlib.pyplot.clf()






