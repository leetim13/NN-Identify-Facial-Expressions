""" Methods for doing logistic regression."""
#Timothy Lee
import numpy as np
from utils import sigmoid
import math

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    # TODO: Finish this function
    b = weights[-1] #bias
    y = sigmoid(np.dot(data, weights[:-1]) + b) #since last element of weight is a bias vector
    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    ce = - np.dot(np.transpose(targets), np.log(y)) #by definition
    frac_correct = (targets == (y > 0.5)).mean() #use 0.5 as threshold, then take fraction
    
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # TODO: Finish this function
    
    y = logistic_predict(weights, data) #after sigmoid transformation

    # TODO: compute f and df without regularization
    f = 0
    for i in range(len(y)): #y is vector of n elements
        f += (-targets[i]*math.log(y[i])) - (1-targets[i])*math.log(1-y[i]) #by formula of L_CE
        
    product = data.T.dot(y-targets)
    df = np.append(product, [[np.sum(y-targets)]], axis = 0) 
    #add vector of sum([y-targets]) to the last row to make df to size (M+1) x 1

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """
    
    w = weights[:-1] #removed bias for weights vector (last element)
    w1 = np.append(w, [[0]], axis=0) #added 0 to match the required M+1 dimension
    f, df, y = logistic(weights, data, targets, hyperparameters) #using logistic()
    
    #regulariizer term = scaled_lambda * (sum of weights^2)
    scaled_lambda = hyperparameters['weight_regularization'] / 2

    f = f + sum(((w1) ** 2) * scaled_lambda) #added regularizer term
    df = df + w1 * hyperparameters['weight_regularization'] #added regularizer term
    
    return f, df, y








