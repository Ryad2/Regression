import numpy as np
from ..utils import label_to_onehot, onehot_to_label


def euclidean_dist(sample, data):
    '''
        Computes eulidean distance.

        Arguments:
            sample (array): sample vector, of shape (D,)
            data (array): training data, of shape (N,D)
        Returns:
            (array): all distances from sample, of shape (N,)
    '''
    return np.sqrt(np.sum((data - sample) ** 2, axis=1))


def kNN_one_sample(sample, data, labels, k, task_kind):
    '''
        Outputs kNN prediction for one sample.

        Arguments:
            sample (array): sample vector, of shape (D,)
            data (array): training data, of shape (N,D)
            labels (array): training labels, of shape (N,)
            k (int): number of nearest neighbors
            task_kind (string): either classification or regression
        Returns:
            (int): the predicted label for the sample vector
    '''
    distances = euclidean_dist(sample, data)
    nn_indices = np.argsort(distances)[:k]
    neighbor_labels = labels[nn_indices]
    if task_kind == "classification":
        best_label = np.argmax(np.bincount(neighbor_labels))
    elif task_kind == "regression":
        best_label = np.mean(neighbor_labels, axis=0)
    return best_label


class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.data = None
        self.labels = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.data = training_data
        self.labels = training_labels

        pred_labels = np.apply_along_axis(kNN_one_sample,
                                          1,
                                          training_data,
                                          self.data,
                                          self.labels,
                                          self.k,
                                          self.task_kind)
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        test_labels = np.apply_along_axis(kNN_one_sample,
                                          1,
                                          test_data,
                                          self.data,
                                          self.labels,
                                          self.k,
                                          self.task_kind)
        return test_labels
