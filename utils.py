# CS6375 Machine Learning
# Gautam Kunapuli
#
# Do not distribute, publish or share this or any other files for this assignment.

import numpy as np

"""
Utilities to load data and organize data sets from the data folder
DO NOT MODIFY THIS FILE.
"""

# The attribute value map is a dictionary of the unique values each attribute can take, that is, it is the
# unique values in each column in the (training) data. For an n x d data matrix (n examples, d features) the
# attribute value map will be of the form:
# { 0: unique values of feature/column 0,
#   1: unique values of feature/column 1,
#   ...
#   d-1: unique values of feature/column d-1 }
#
# self.attribute_value_map = {i: np.unique(self.examples['train'][:, i]) for i in range(self.shape['train'][1])}


class DataSet:
    def __init__(self, data_set):
        """
        Initialize a data set and load both training and test data
        DO NOT MODIFY THIS FUNCTION
        """
        self.name = data_set

        # The training and test labels
        self.labels = {'train': None, 'test': None}

        # The training and test examples
        self.examples = {'train': None, 'test': None}

        # Load all the data for this data set
        for data in ['train', 'test']:
            self.load_file(data)

        # The shape of the training and test data matrices
        self.num_train = self.examples['train'].shape[0]
        self.num_test = self.examples['test'].shape[0]
        self.dim = self.examples['train'].shape[1]

    def load_file(self, dset_type):
        """
        Load a training set of the specified type (train/test). Returns None if either the training or test files were
        not found. NOTE: This is hard-coded to use only the first seven columns, and will not work with all data sets.
        DO NOT MODIFY THIS FUNCTION
        """
        path = './data/{0}.{1}'.format(self.name, dset_type)
        try:
            file_contents = np.genfromtxt(path, missing_values=0, skip_header=0, delimiter=',', dtype=int)

            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))


if __name__ == '__main__':
    d = DataSet('monks-1')
    print(d)
