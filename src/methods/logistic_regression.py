import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters = 500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None



    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###
        def softmax(weights):
            exp = np.exp(training_data @ weights)
            return exp / np.sum(exp, axis=0)

        def gradient(weights):
            return training_data.T @ (softmax(weights) - training_labels)

        weights = np.random.normal(0, 0.1, (training_data.shape[1], training_labels.shape[1]))
        for it in range(self.max_iters):
            weights = weights - self.lr * gradient(weights)

        self.weights = weights
        pred_labels = onehot_to_label(softmax(training_data, self.weights))
        ###
        ##
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        predict_labels = onehot_to_label(softmax(test_data, self.weights))
        ###
        ##
        return predict_labels
