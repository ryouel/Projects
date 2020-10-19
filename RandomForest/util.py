from scipy import stats
import numpy as np


# This method computes entropy for information gain
def entropy(class_y):
    # Inputs:
    #   class_y  : data containing all labels

    #Compute the entropy for a list of classes
    try:
        np.seterr(all='raise')
        zeros = 0
        ones = 0
        for y in class_y:
            if y == 0:
                zeros += 1
            else:
                ones += 1

        entropy = -(zeros/len(class_y))*np.log2(zeros/len(class_y)) - (ones/len(class_y))*np.log2(ones/len(class_y))
    except:
        entropy = 0
    return entropy


def partition_classes(X, y, split_attribute, split_val):
    # Inputs:
    #   X               : data containing all attributes
    #   y               : labels
    #   split_attribute : column index of the attribute to split on
    #   split_val       : either a numerical or categorical value to divide the split_attribute
    
    # Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
    X_left = []
    X_right = []
    
    y_left = []
    y_right = []

    try:
        x = int(split_val)
        for attr, lbl in zip(X,y):
            if attr[split_attribute] <= split_val:
                X_left.append(attr)
                y_left.append(lbl)
            else:
                X_right.append(attr)
                y_right.append(lbl)
    except:
        print('category')
        for attr, lbl in zip(X,y):
            if attr[split_attribute] == split_val:
                X_left.append(attr)
                y_left.append(lbl)
            else:
                X_right.append(attr)
                y_right.append(lbl)
    return (X_left, X_right, y_left, y_right)

    
def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value
    
    # Compute and return the information gain from partitioning the previous_y labels
    # into the current_y labels.
    total = 0
    for y in current_y:
        total +=  entropy(y) * len(y) / len(previous_y)
    info_gain = entropy(previous_y) - total
    return info_gain
    
