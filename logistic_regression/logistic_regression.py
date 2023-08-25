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
        pass

    def h(self, x):
        return sigmoid(self.weights.T @ x.T)

    def p(self, X, y):
        return np.exp(self.h(X), y) @ np.exp(1 - self.h(X), 1-y)

    def p_i(self, X, y, i):
        return self.p(np.exp(X,i), np.exp(y,i))

    def L(self, X, y, n):
        return np.prod(self.p_i(X,y,n))

    def l(self, X, y, n):
        return np.log(self.L(X, y, n))

    def gradient_ascent(self, X, y, epsilon, n):
        sum = np.zeros((1,2))

        for i in range(1, n+1):
            sum += (y**i - self.h(X**i)) @ (X**i)

        self.weights = (self.weights.T + epsilon * sum).T
        #print(self.weights)
        return self.weights


    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        print(X.shape, y.shape)
        learning_rate = 0.001
        for i in range(10):
            self.gradient_ascent(X, y, learning_rate, len(y))
            loss = binary_cross_entropy(y, self.predict(X))
            #print(loss)



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

        