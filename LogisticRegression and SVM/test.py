import numpy as np

# test1 = np.ones([2,3])
# test2 = np.ones([2,1])
#
# test2[0] = 2
#
# print(np.sum(np.multiply(test2,test1),keepdims=True,axis=0))


# loading the data
X_train_buf = np.loadtxt('classification/train_features.dat',dtype=float,delimiter=' ')
Y_train = np.loadtxt('classification/train_labels.dat',dtype=float,delimiter=' ').astype(int)
Y_train = np.reshape(Y_train,[Y_train.shape[0],1])
X_test_buf = np.loadtxt('classification/test_features.dat',dtype=float,delimiter=' ')
Y_test = np.loadtxt('classification/test_labels.dat',dtype=float,delimiter=' ').astype(int)
Y_test = np.reshape(Y_test, [Y_test.shape[0], 1])

X_train = np.ones([X_train_buf.shape[0],X_train_buf.shape[1]+1])
X_train[:,1:] = X_train_buf
X_test = np.ones([X_test_buf.shape[0],X_test_buf.shape[1]+1])
X_test[:,1:] = X_test_buf




