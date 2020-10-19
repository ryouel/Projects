from decision_tree import DecisionTree
import csv
import numpy as np 
import ast
import time

class RandomForest(object):
    num_trees = 0
    decision_trees = []

    # the bootstrapping datasets for trees
    bootstraps_datasets = []

    # the true class labels, corresponding to records in the bootstrapping datasets
    bootstraps_labels = []

    def __init__(self, num_trees):
        # Initialization done here
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree() for i in range(num_trees)]

    def _bootstrapping(self, XX, n):
        # Create a sample dataset of size n by sampling with replacement from the original dataset XX.

        samples = []  # sampled dataset
        labels = []  # class labels for the sampled records

        #create array with randomply sampled indicies
        sample_ind = np.random.choice(len(XX), n, replace=True)
        
        #store the samples/labels
        for i in sample_ind:
            r = XX[i]
            s = r[:-1]
            l = r[-1]
            samples.append(s)
            labels.append(l)

        return (samples, labels)

    def bootstrapping(self, XX):
        # Initializing the bootstap datasets for each tree
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):
        # Train `num_trees` decision trees using the bootstraps datasets
        # and labels by calling the learn function from DecisionTree class.
        for i in range(self.num_trees):
            self.decision_trees[i].learn(self.bootstraps_datasets[i],self.bootstraps_labels[i])
        pass

    def voting(self, X):
        y = []

        for record in X:
            # Following steps have been performed here:
            #   1. Find the set of trees that consider the record as an
            #      out-of-bag sample.
            #   2. Predict the label using each of the above found trees.
            #   3. Use majority vote to find the final label for this recod.
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)

            counts = np.bincount(votes)

            if len(counts) == 0:
                # Special case
                #  Handle the case where the record is not an out-of-bag sample
                #  for any of the trees.

                #store the prediction from the last tree
                y = np.append(y,self.decision_trees[len(self.bootstraps_datasets) - 1].classify(record))
            else:
                y = np.append(y, np.argmax(counts))

        return y


# DO NOT change the main function apart from the forest_size parameter!
def main():
    start = time.time()
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels
    numerical_cols = set([i for i in range(0, 9)])  # indices of numeric attributes (columns)

    # Loading data set
    print("reading pulsar_stars")
    with open("pulsar_stars.csv") as f:
        next(f, None)
        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                if i in numerical_cols:
                    xline.append(ast.literal_eval(line[i]))
                else:
                    xline.append(line[i])

            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])

    # TODO: Initialize according to your implementation
    # VERY IMPORTANT: Minimum forest_size should be 10
    forest_size = 10

    # Initializing a random forest.
    randomForest = RandomForest(forest_size)

    # Creating the bootstrapping datasets
    print("creating the bootstrap datasets")
    randomForest.bootstrapping(XX)

    # Building trees in the forest
    print("fitting the forest")
    randomForest.fitting()

    # Calculating an unbiased error estimation of the random forest
    # based on out-of-bag (OOB) error estimate.
    y_predicted = randomForest.voting(X)

    # Comparing predicted and true labels
    results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))

    print("accuracy: %.4f" % accuracy)
    print("OOB estimate: %.4f" % (1 - accuracy))
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()