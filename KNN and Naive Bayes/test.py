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


x1 = np.array([3,0,2])
x2 = np.arange(12.0).reshape((4, 3))
# print(x1,x2)
# print(x2[:,0])
# print(np.sum(x2,axis=1))
# print(x1-x2)
print(x1.argsort()[:3])
# print(np.divide(x1, x2))
# print(x2-x1)
#
# for sample in x2:
#     print(sample)
#     break

# x = [[20,1],[10,2]]
# x.append([30,5])
#
# print(sorted(x))

