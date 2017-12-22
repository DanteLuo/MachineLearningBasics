from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from csv import reader
import numpy as np
import pickle


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


def StockRandomForest(args, n_fold = 5, verbose=False):
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

    X = np.asarray(X)

    y = load_csv("Label.csv")
    y = np.reshape(y[1:],(len(y)-1,))

    # K-fold validation
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    max_depth = args['max_depth']
    n_estimators = args['n_estimators']
    # max_features = args['max_features']
    # min_samples_split = args['min_samples_split']
    # if min_samples_split > 1:
    #     min_samples_split = int(min_samples_split)
    # elif min_samples_split <= 0.0:
    #     min_samples_split =

    # bootstrap = args['bootstrap']
    # criterion = args['tree_criterion']
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                 max_features=None, criterion='entropy',
                                 bootstrap=True, n_jobs=-1)

    train_accuracy = 0.
    test_accuracy = 0.

    for train_id, test_id in kf.split(X):
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]

        # fit model
        clf.fit(X_train,y_train)
        train_accuracy += clf.score(X_train,y_train)*100/n_fold
        test_accuracy += clf.score(X_test,y_test)*100/n_fold

    if verbose:
        print(clf.feature_importances_)
        print('The params are:', clf.get_params(deep=True))
        print("The prediction accuracy is",test_accuracy)
        print("The training accuracy is ",train_accuracy)

    return 100-test_accuracy, 100-train_accuracy

def running_experiment(args, verbose=False):
    max_depth_arr = args['max_depth']
    n_estimators_arr = args['n_estimators']
    test_errors = np.zeros([len(max_depth_arr),len(n_estimators_arr)])
    train_errors = np.zeros([len(max_depth_arr),len(n_estimators_arr)])

    for id_depth, max_depth in enumerate(max_depth_arr):
        for id_estimator, n_estimators in enumerate(n_estimators_arr):
            test_error, train_error = StockRandomForest({'max_depth':max_depth,'n_estimators':n_estimators},verbose=verbose)
            test_errors[id_depth, id_estimator] = test_error
            train_errors[id_depth, id_estimator] = train_error

    with open('accuracy.test','wb') as f:
        pickle.dump((max_depth_arr,n_estimators_arr,test_errors,train_errors),f)

def plotter(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f)
        X = data[0]
        Y = data[1]
        Z1 = data[2]
        Z2 = data[3]

    print(X.shape,Y.shape)
    print(Z1.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Y, X = np.meshgrid(Y,X)
    print(X.shape,Y.shape)
    surf1 = ax.plot_surface(X, Y, Z1, color=[0,1,0,0.7])
    surf2 = ax.plot_surface(X, Y, Z2, color=[1,0,0,0.7])
    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", c=[0,1,0], marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    ax.set_xlabel('Maximum depth')
    ax.set_ylabel('Tree size')
    ax.set_zlabel('% Error')
    ax.legend([fake2Dline1,fake2Dline2],['Test error','Training error'],numpoints=1)
    ax.view_init(5,60)
    plt.show()

def main():
    # StockRandomForest({'max_depth':40,'n_estimators':1}, verbose=True)
    # import timeit
    # start_time = timeit.default_timer()
    # max_depth_arr = np.arange(20,200,20)
    # n_estimators_arr = np.arange(1,20,4)
    # running_experiment({'max_depth':max_depth_arr,'n_estimators':n_estimators_arr})
    # print('The time of experiment is', timeit.default_timer() - start_time)
    plotter('accuracy.test')
    # print('The time of execution is', timeit.default_timer() - start_time)

if __name__ == '__main__':
    main()
