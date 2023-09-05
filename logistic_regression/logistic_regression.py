import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:

    def __init__(self, m = 2):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        # Weights that will be used in training
        self.weights = np.ones((m,1))

    def h(self, x):
        return sigmoid(self.weights.T @ x.T)


    def gradient_descent(self, X, y, epsilon):
        """
        Gradient descent computed using numpy to maximize efficiency
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing
                m binary 0.0/1.0 labels
            epsilon: Learning rate

        Returns:
        Weights
        """
        sum = 2 * np.sum(self.h(X) - y, axis=0, keepdims=True) @ X
        self.weights = (self.weights.T - epsilon * sum).T
        return self.weights


    def fit(self, X, y, lr = 0.0005, iterations = 1000):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        for i in range(iterations):
            self.gradient_descent(X, y, lr)



    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        return self.h(X).flatten()

def preprocess_mirror(data, shift_value = 0.5):
    # Data is scattered with 0 in middle and 1s at each end
    # Change the range from [0,1] to [-0.5,0.5] so that we can abs the 0s to split them
    # and then the different data will have a clear split
    abs_data = np.abs(data - shift_value)
    return feature_scale(abs_data)

def feature_scale(data):
    # Standardization

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        