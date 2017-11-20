from sklearn.ensemble import RandomForestClassifier
from csv import reader
import numpy as np
import random

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


def StockRandomForest(args, verbose=False):
    # loading data
    filename = "data.csv"
    X = load_csv(filename)
    X = X[1:]

    # convert string attributes to integers
    for i in range(0, len(X[0])):
        str_column_to_float(X, i)

    day_count = 1
    for i in range(0, len(X)):
        X[i].append(day_count)
        day_count += 1
        if day_count == 366:
            day_count = 1

    y = load_csv("Label.csv")
    y = np.reshape(y[1:],(len(y)-1,))

    train_portion = int(len(X)*9/10)

    train_id = random.sample(range(0,len(X)-1),train_portion)
    X_train = list()
    y_train = list()

    for i in train_id:
        X_train.append(X[i])
        y_train.append(y[i])

    test_id = list(set(range(0,len(X)-1))-set(train_id))
    X_test = list()
    y_test = list()

    for i in test_id:
        X_test.append(X[i])
        y_test.append(y[i])

    max_depth = args['max_depth']
    n_estimators = args['n_estimators']

    # fit model
    clf = RandomForestClassifier(random_state=0, max_depth=max_depth, n_estimators=n_estimators)
    # clf = RandomForestClassifier(random_state=0)

    clf.fit(X_train,y_train)

    # feature importances
    if verbose:
        print(clf.feature_importances_)
        print('The params are:',clf.get_params(deep=True))

    # calculate prediction error
    y_predict_test = clf.predict(X_test)
    test_accuracy = 100 - np.count_nonzero(np.asarray(y_predict_test,dtype=int)-np.asarray(y_test,dtype=int))/float(len(y_test))*100

    y_predict_train = clf.predict(X_train)
    train_accuracy = 100 - np.count_nonzero(np.asarray(y_predict_train,dtype=int)-np.asarray(y_train,dtype=int))/float(len(y_train))*100

    if verbose:
        print("The prediction accuracy is",test_accuracy)
        print("The training accuracy is ",train_accuracy)

    return 100-test_accuracy

def main():
    StockRandomForest({'max_depth':None,'n_estimators':10})

if __name__ == '__main__':
    main()
