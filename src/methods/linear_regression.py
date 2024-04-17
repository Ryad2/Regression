import numpy as np


class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.weights = None
        self.lmda = lmda
        self.task_kind = 'regression'

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        #Use the pseudo-inverse if no regularizer as the matrix might not be invertible
        if self.lmda == 0:
            self.weights = \
                (np.linalg.pinv(training_data) @ training_labels)
        else:
            self.weights = \
                (np.linalg.inv(training_data.T @ training_data + self.lmda * np.eye(training_data.shape[1]))
                 @ training_data.T @ training_labels)

        pred_regression_targets = training_data @ self.weights

        return pred_regression_targets

    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        pred_regression_targets = test_data @ self.weights
        return pred_regression_targets
