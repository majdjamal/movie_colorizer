
__author__ = 'Majd Jamal'

import numpy as np
import matplotlib.pyplot as plt

X = np.load('X.npy')
Y = np.load('Y.npy')


Npts, _,_,_ = X.shape

indicies = np.arange(Npts)
np.random.shuffle(indicies)

train_npts = int(0.7 * Npts)
train_po = indicies[:train_npts]

val_npts = int(0.15 * Npts)
val_po = indicies[train_npts: train_npts + val_npts]

test_po = indicies[train_npts + val_npts:]

X_train = X[train_po]
y_train = Y[train_po]

X_val = X[val_po]
y_val = Y[val_po]

X_test = X[test_po]
y_test = Y[test_po]

plt.imshow(X_val[0])
plt.show()

np.save('processed_data/X_train.npy', X_train)
np.save('processed_data/y_train.npy', y_train)

np.save('processed_data/X_val.npy', X_val)
np.save('processed_data/y_val.npy', y_val)

np.save('processed_data/X_test.npy', X_test)
np.save('processed_data/y_test.npy', y_test)
