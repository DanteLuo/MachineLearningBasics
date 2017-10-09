import numpy as np
import scipy.io as sio
from collections import Counter
import matplotlib.pyplot as plt


# classify the sample by K-Nearest Neighbors and break tie with random choice
def knn_error_checking(train_data,test_data,k):
    X_train = train_data[:,1:-1]
    Y_train = train_data[:,0]
    X_test = test_data[:,1:-1]
    Y_test = test_data[:,0]
    test_predict = np.zeros(test_data.shape[0])

    for sample_id, sample in enumerate(X_test):
        # l2 norm
        # distance = np.sqrt(np.sum(np.square(sample - X_train),axis=1))
        # l1 norm
        distance = np.abs(np.sum((sample - X_train),axis=1))
        prediction = Y_train[distance.argsort()[:k]]
        test_predict[sample_id] = Counter(prediction).most_common(1)[0][0]

    # print out the training error
    training_err = float(np.count_nonzero(test_predict-Y_test))/float(Y_test.shape[0])*100

    return training_err


def main():
    # load the data
    mnist_data = sio.loadmat('mnist_data.mat')
    # print(mnist_data)
    train_data = np.asarray(mnist_data['train'])
    test_data = np.asarray(mnist_data['test'])
    # print(train_data)
    print('The size of train data is:',train_data.shape,'and the test data is:',test_data.shape)

    # random sample N test images
    sample_n = 100
    choices = range(test_data.shape[0])
    np.random.seed(32)

    # prediction and calculating training error
    num_avg = 5
    k_candidates = [1, 5, 9, 13]
    training_err = np.zeros([num_avg,len(k_candidates)])
    for num_episode in range(num_avg):
        sample_index = np.random.choice(choices, sample_n)
        test_sample = test_data[sample_index][:]
        for k_ind, k in enumerate(k_candidates):
            training_err[num_episode][k_ind] = knn_error_checking(train_data,test_sample,k)

    # averaging the error for a steady performance
    training_err_avg = np.mean(training_err,axis=0)
    plt.plot(k_candidates,training_err_avg,'or')
    plt.plot(k_candidates,training_err_avg,'b')
    plt.ylabel('Training error %')
    plt.xlabel('K number')
    plt.show()


if __name__ == '__main__':
    main()

