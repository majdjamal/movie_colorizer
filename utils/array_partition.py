
__author__ = 'Majd Jamal'

import numpy as np
import matplotlib.pyplot as plt

def array_partition(training_part: float = 0.7, validation_part: float = 0.15) -> None:
    """ Partition data into training 70%, validation 15% and test set 15%.
    :params training_part: Training fraction
    :params validation_part: Validation fraction
    """

    X = np.load('X.npy')
    Y = np.load('Y.npy')


    Npts, _,_,_ = X.shape

    indicies = np.arange(Npts)
    np.random.shuffle(indicies)

    train_npts = int(training_part * Npts)
    train_po = indicies[:train_npts]

    val_npts = int(validation_part * Npts)
    val_po = indicies[train_npts: train_npts + val_npts]

    test_po = indicies[train_npts + val_npts:]

    X_train = X[train_po]
    y_train = Y[train_po]

    X_val = X[val_po]
    y_val = Y[val_po]

    X_test = X[test_po]
    y_test = Y[test_po]


    np.save('processed_data/X_train.npy', X_train)
    np.save('processed_data/y_train.npy', y_train)

    np.save('processed_data/X_val.npy', X_val)
    np.save('processed_data/y_val.npy', y_val)

    np.save('processed_data/X_test.npy', X_test)
    np.save('processed_data/y_test.npy', y_test)
