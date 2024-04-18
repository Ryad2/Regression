import numpy as np

from ..utils import label_to_onehot, onehot_to_label


def softmax(data, weights):
    '''
        Computes softmax function.

        Arguments:
            data (array): training data, of shape (N,D)
            weights (array): model weights, of shape (D,nb of classes)
        Returns:
            (array): output of softmax function, of shape (N,nb of classes)
    '''
    exp = np.exp(data @ weights)
    return exp / (np.sum(exp, axis=1).reshape(-1, 1))


def gradient_logistic_regression(data, weights, one_hot_labels):
    '''
    Computes the gradient for logistic regression gradient descent.

    Arguments:
        data (array): training data, of shape (N,D)
        weights (array): model weights, of shape (D,nb of classes)
        one_hot_labels (array): training labels in one hot encoding, of shape (N,nb of classes)
    Returns:
        (array): gradient, of shape (D,nb of classes)
    '''
    return data.T @ (softmax(data, weights) - one_hot_labels)


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, optimizer = 'SGD',task_kind = 'classification'):
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
        self.optimizer = optimizer
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        def sgd(one_hot_labels, weights):
            for it in range(self.max_iters):
                weights -= self.lr * gradient_logistic_regression(training_data, weights, one_hot_labels)
            
            self.weights = weights
            return onehot_to_label(softmax(training_data, self.weights))
        
        def adam(one_hot_labels, weights):
            # Initialize Adam parameters
            m = np.zeros_like(weights)
            v = np.zeros_like(weights)
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8

            for it in range(self.max_iters):
                gradient = gradient_logistic_regression(training_data, weights, one_hot_labels)
                # Update first moment estimate
                m = beta1 * m + (1 - beta1) * gradient
                # Update second moment estimate
                v = beta2 * v + (1 - beta2) * (gradient ** 2)
                # bias-corrected first and second moment estimates
                m_hat = m / (1 - beta1 ** (it + 1))
                v_hat = v / (1 - beta2 ** (it + 1))
                # Update weights using Adam update rule
                new_weights = weights - self.lr * m_hat / (np.sqrt(v_hat) + epsilon)
                if np.allclose(new_weights, weights):
                    break
                else:
                    weights = new_weights

            self.weights = weights
            return onehot_to_label(softmax(training_data, self.weights))

        one_hot_labels = label_to_onehot(training_labels)
        # Xavier/Glorot weights initialization
        std = np.sqrt(2.0 / (training_data.shape[1] + one_hot_labels.shape[1]))
        weights = np.random.normal(0, std, (training_data.shape[1], one_hot_labels.shape[1]))

        if self.optimizer == 'ADAM':
            pred_labels = adam(one_hot_labels, weights)
        else:
            pred_labels = sgd(one_hot_labels, weights)
        
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        predict_labels = onehot_to_label(softmax(test_data, self.weights))

        return predict_labels
