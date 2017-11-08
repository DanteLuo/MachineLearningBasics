import numpy as np
from csv import reader
from sklearn.svm import LinearSVC as LSVC
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# Load a CSV file
def load_csv(filename):
    #init the dataset as a list
	dataset = list()
    #open it as a readable file
	with open(filename, 'r') as file:
        #init the csv reader
		csv_reader = reader(file)
        #for every row in the dataset
		for row in csv_reader:
            #add that row as an element in our dataset list (2D Matrix of values)
			dataset.append(row)
    #return in-memory data matrix
	return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    #iterate throw all the rows in our data matrix
	for row in dataset:
        #for the given column index, convert all values in that column to floats
		row[column] = float(row[column].strip())


# loading data
filename = "diabetes_scale.csv"
data = load_csv(filename)

# convert string attributes to integers
for i in range(0, len(data[0])):
    str_column_to_float(data, i)

data = np.asarray(data)
X = data[:,1:]
y = np.asarray(data[:,0],dtype=int)

# generate 5-fold data
is_verbose = 0
n_fold = 5
kf = KFold(n_splits=n_fold,shuffle=True,random_state=42)

# soft-margin svm
c_candidates = np.linspace(0.1,2.,20)
err = np.zeros([c_candidates.shape[0],2],dtype=float)

curr_c = 0
for c in c_candidates:
    curr_fold = 0
    err_test = 0
    err_train = 0
    for train_id, test_id in kf.split(X):
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]

        lsvm = LSVC(penalty='l2', loss='hinge', C=c, verbose=is_verbose, random_state=42, max_iter=1000)
        lsvm.fit(X_train, y_train)
        y_predict_test = lsvm.predict(X_test)
        y_predict_train = lsvm.predict(X_train)
        err_test += np.count_nonzero(y_predict_test - y_test) / float(y_test.shape[0]) * 100
        err_train += np.count_nonzero(y_predict_train - y_train) / float(y_train.shape[0]) * 100

    err[curr_c, 1] = err_test/n_fold
    err[curr_c, 0] = err_train/n_fold
    curr_c += 1

print('The best C achieves the accuracy is',c_candidates[np.argmin(err[:,1])],'and the test error is',np.min(err[:,1]))

# hard-margin svm
c = 1e6
err_test_hard = 0.
err_train_hard = 0.

for train_id, test_id in kf.split(X):
    X_train, X_test = X[train_id], X[test_id]
    y_train, y_test = y[train_id], y[test_id]

    lsvm = LSVC(penalty='l2', loss='hinge', C=c, verbose=is_verbose, random_state=42, max_iter=1000)
    lsvm.fit(X_train, y_train)
    y_predict_test = lsvm.predict(X_test)
    y_predict_train = lsvm.predict(X_train)
    err_test_hard += np.count_nonzero(y_predict_test - y_test) / float(y_test.shape[0]) * 100
    err_train_hard += np.count_nonzero(y_predict_train - y_train) / float(y_train.shape[0]) * 100

err_test_hard = err_test_hard/n_fold
err_train_hard = err_train_hard/n_fold

print('The test error of hard-margin svm is',err_test_hard,'and its training error is',err_train_hard)


# visualize error
plt.plot(c_candidates,err[:,0],'-or',c_candidates,err[:,1],'-ob')
plt.ylabel('Error')
plt.xlabel('C')
plt.legend(['Training error','Test error'])
plt.show()