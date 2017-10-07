import os
import csv
import numpy as np


# y = list()
#
# full_path = os.path.expanduser('~/PycharmProjects/ML/HW_two/spam_classification/spam_classification/SPARSE_TRAIN.csv')
# with open(full_path, newline='') as train:
#     reader = csv.reader(train, delimiter=' ')
#     for i, row in enumerate(reader):
#         print(i,row)
#         break


# test for the np.sum function
# test = np.ones([3,4])
# test[1][2] = 54
#
# label = np.asarray([1,0,1])
# print(np.argwhere(label>0))
#
# print(test)
# test_one = np.sum(test[np.argwhere(label>0)][:],axis=0)[0]
# test_two = np.sum(test_one)
# print(test_one)
# print(test_two)


x1 = np.array([1,0,2])
x2 = np.arange(3.0).reshape((1, 3))
print(x1[x1.argsort()[-3:][0]])
print(np.divide(x1, x2))



