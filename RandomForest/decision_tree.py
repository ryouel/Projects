from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {}
        pass

    def learn(self, X, y):
        #Train the decision tree (self.tree) using the the sample X and labels y

        #test if y is the same
        y_sum = np.sum(y)
        if y_sum == len(y):
            self.tree['output'] = 1
            return self.tree
        elif y_sum == 0:
            self.tree['output'] = 0
            return self.tree

        #test if x is the same
        x_same = 1
        for i in range(len(X) - 1):
            if X[i] == X[-1]:
                x_same += 1
        if x_same == (len(X) - 1):
            self.tree['output'] = y_sum // len(y)
            return self.tree

        #compute information gain of each feature and split on best one
        split_attribute = 0
        info_gain = []
        split_vals = []
        for col in zip(*X):
            #split based on average
            split_val = np.average(col)
            split_vals.append(split_val)

            #split and compute information gain
            X_left, X_right, y_left, y_right = partition_classes(X, y, split_attribute, split_val)
            current_y = []
            current_y.append(y_left)
            current_y.append(y_right)
            ig = information_gain(y, current_y)
            info_gain.append(ig)
            
            #increment the column number
            split_attribute += 1

        #find highest info gain and split the tree there
        max_val = max(info_gain)
        max_ind = info_gain.index(max_val)
        split_val = split_vals[max_ind]  
        X_left, X_right, y_left, y_right = partition_classes(X, y, max_ind, split_val)

        #store the column and value to split on for classifying and set output -1 to flag it is not a leaf node
        self.tree['split col'] = max_ind
        self.tree['split val'] = split_val
        self.tree['output'] = -1

        #right and left subtrees
        self.tree['left'] = DecisionTree().learn(X_left, y_left)
        self.tree['right'] = DecisionTree().learn(X_right, y_right)

        #return the tree
        return self.tree

    def classify(self, record):
        #classify the record using self.tree and return the predicted label
        #set the tree pointer to the whole tree and loop until leaf node is reached
        curr_tree = self.tree
        while curr_tree['output'] == -1:
            #obtain splitting values to traverse to right tree
            split_attribute = curr_tree['split col']
            split_val = curr_tree['split val']

            #update the tree pointer to the correct subtree
            if record[split_attribute] <= split_val:
                curr_tree = curr_tree['left']
            else:
                curr_tree = curr_tree['right']
        
        return curr_tree['output']