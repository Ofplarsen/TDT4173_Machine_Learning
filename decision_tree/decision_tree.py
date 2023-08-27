import random

import numpy as np
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)



class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.root = {}
        self.sorted_names = []

    def p(self, t, n):
        return t/n

    def entropy(self, S):
        entropy = 0
        for i in S:
            p_i = self.p(i, sum(S))
            if p_i == 0:
                continue
            entropy -= p_i * np.log2(p_i)

        return entropy

    def A(self, A, y):
        A_y = pd.concat([A, y], axis=1)

        split = A_y.groupby(A_y.columns[0])[A_y.columns[1]].value_counts().unstack(fill_value=0)
        return [[count_yes, count_no] for count_yes, count_no in split.values] # Generalise more

    def gain(self, S, A):
        total = 0
        for s_v in A:
            #print(s_v)
            total += sum(s_v) / sum(S) * self.entropy(s_v)

        return self.entropy(S) - total
    
    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        value_counts = y.value_counts()
        if len(value_counts) == 1:
            self.root['y'] = value_counts.index.tolist()[0]
            return

        entropy = self.entropy(value_counts.values)

        dfs = []
        A = []
        gain = []
        for c in X.columns:
            #print(c)
            A = self.A(X[c], y)
            #print(A)

            #print(self.gain(value_counts, A))
            gain.append(self.gain(value_counts, A))

        #print(gain)
        #print(X.columns.tolist())

        sorted_pairs = sorted(zip(gain, X.columns.tolist()), reverse=True)

        # Unpack the sorted pairs into separate lists
        sorted_names = [name for _, name in sorted_pairs]
        #print(sorted_names)

        self.sorted_names = sorted_names

        X = X[sorted_names]

        for index, i in enumerate(X.values):
            tree = self.root

            for x in i:
                if x not in tree:
                    tree[x] = {}
                tree = tree[x]

            if 'y' not in tree:
                tree['y'] = y[index]



        print(pd.Series(self.root))


    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        y = []
        X = X[self.sorted_names]
        for index, i in enumerate(X.values):
            tree = self.root
            for x in i:
                if x not in tree:
                    x = random.choice(list(tree.keys()))
                tree = tree[x]
            y.append(tree['y'])

        return np.array(y)



    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        rules = []
        for key, value in self.root.items():
            rules.append(([(key, value)], 'yes'))
            print(f"Key: {key}, Value: {value}")
        return rules

# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



