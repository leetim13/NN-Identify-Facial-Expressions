import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt

def run_logistic_regression(weight_regularization):
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()
    N, M = train_inputs.shape
    # N is number of examples; M is the number of features per example.
    # TODO: Set hyperparameters

    hyperparameters = {
                    'learning_rate': 0.01,
                    'weight_regularization': weight_regularization,
                    'num_iterations': 50
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.randn(M + 1, 1) /10

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    ce_training = []
    ce_validation = []
    ce_testing = []

    fc_training = []
    fc_validation = []
    fc_testing = []

    # Begin learning with gradient descent
    for t in range(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
#        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)


        testing_prediction = logistic_predict(weights, test_inputs)
        cross_entropy_test, frac_correct_test = evaluate(test_targets, testing_prediction)
        print("\n testing set cross entropy: ", str(cross_entropy_test), "testing set frac_correct ", frac_correct_test, " amd "
                                                                                                                         "lambda ",weight_regularization)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        print("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}".format(
                   t+1, float(f / N), float(cross_entropy_train), float(frac_correct_train*100),
                   float(cross_entropy_valid), float(frac_correct_valid*100)))


        if t == (hyperparameters['num_iterations'] - 1):

        # Calculate the classification error:
            frac_error_train = 1 - frac_correct_train
            frac_error_valid = 1 - frac_correct_valid
            frac_error_test = 1 - frac_correct_test

        ce_training = np.append(ce_training, cross_entropy_train)
        ce_validation = np.append(ce_validation, cross_entropy_valid)

        fc_training = np.append(fc_training, frac_correct_train)
        fc_validation = np.append(fc_validation, frac_correct_valid)


    # plt.plot(ce_training, label = "training set")
    # plt.plot(ce_validation, label = "validation set")
    # plt.title("Cross Entropy with small training set using \n learning rate "+ str(hyperparameters['learning_rate'])+
    #           " and " + str(hyperparameters['num_iterations']) + " iterations with lambda " + str(weight_regularization))
    # plt.legend()
    # plt.show()
    # plt.savefig("ce" + str(weight_regularization) + ".png")
    # plt.title("Fraction Correctness with training set using \n learning rate "+ str(hyperparameters['learning_rate'])+
    #           " and " + str(hyperparameters['num_iterations']) + " iterations with lambda "+ str(weight_regularization))
    # plt.plot(fc_training, label = "training set")
    # plt.plot(fc_validation, label = "validation set")
    # plt.legend()
    # plt.show()

    p_training = logistic_predict(weights, train_inputs)
    cross_entropy_train, frac_correct_train = evaluate(train_targets, p_training)
    predictions_valid = logistic_predict(weights, valid_inputs)
    cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
    return cross_entropy_train, cross_entropy_valid, frac_correct_train, frac_correct_valid

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print ("diff =", diff)

if __name__ == '__main__':
    average_train_ce = []
    average_train_fc = []
    average_valid_ce = []
    average_valid_fc = []

    sum_train_ce = 0
    sum_train_fc = 0
    sum_valid_ce = 0
    sum_valid_fc = 0
    lambds = [0, 0.001, 0.01, 0.1, 1.0]
    for lambd in lambds:
        run_logistic_regression(weight_regularization=lambd)

        for i in range(5):
            training_ce, validation_ce, training_fc, validation_fc = run_logistic_regression(weight_regularization=lambd)
            sum_train_ce += training_ce
            sum_train_fc += training_fc
            sum_valid_ce += validation_ce
            sum_valid_fc += validation_fc
        average_train_ce.append(float(sum_train_ce)/5)
        average_train_fc.append(float(sum_train_fc)/5)
        average_valid_ce.append(float(sum_valid_ce)/5)
        average_valid_fc.append(float(sum_valid_fc)/5)
        sum_train_ce = 0
        sum_train_fc = 0
        sum_valid_ce = 0
        sum_valid_fc = 0


    plt.plot(lambds, average_train_ce, label="train")
    plt.plot(lambds, average_valid_ce, label="valid")
    plt.xlabel("lambda values (penalty)")
    plt.ylabel("average cross entropy")
    plt.title("Plot of average cross entropy vs lambda values")
    plt.legend()
    plt.show()

    plt.plot(lambds, average_train_fc, label="train")
    plt.plot(lambds, average_valid_fc, label="valid")
    plt.xlabel("lambda values (penalty)")
    plt.ylabel("average fractional correctness")
    plt.title("Plot of classification error vs lambda values")
    plt.legend()
    plt.show()




#    run_logistic_regression(weight_regularization=1)



























