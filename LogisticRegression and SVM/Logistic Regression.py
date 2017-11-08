import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1./(1.+np.exp(-x))


def loss(x,y,w):
    h_x = sigmoid(x.dot(w))
    return np.sum(-np.multiply(y,np.log(h_x))-np.multiply((1.-y),np.log(1.-h_x)))


def gradient(x,y,w):
    h_x = sigmoid(x.dot(w))
    return np.transpose(np.sum(np.multiply((h_x-y),x),axis=0,keepdims=True))


def Hessian(x,w):
    h_x = sigmoid(x.dot(w))
    H = np.diag(np.reshape(np.multiply(h_x,(1.-h_x)),[h_x.shape[0],]))
    x_T = np.transpose(x)
    return x_T.dot(H).dot(x) # 3x900x900x900x900x3


def logistic_regression(X_train, Y_train, epsilon, X_test, Y_test):
    w_T = np.zeros([X_test.shape[1],1],dtype=float)
    dLoss = epsilon + 1.
    Loss_train = list()
    Loss_test = list()
    num_steps = 0

    while dLoss>=epsilon:

        grad_L = gradient(X_train,Y_train,w_T)
        H_L = Hessian(X_train,w_T)
        H_inv_L = np.linalg.inv(H_L)
        Loss_prev = loss(X_train,Y_train,w_T)

        # update w
        w_T -= H_inv_L.dot(grad_L)
        Loss_after = loss(X_train,Y_train,w_T)
        dLoss = np.abs(Loss_after-Loss_prev)

        # calculate training err and test err
        Loss_train.append(Loss_after)
        Loss_test.append(loss(X_test,Y_test,w_T))

        num_steps += 1

    return w_T,Loss_train,Loss_test,num_steps


def main():

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

    w,Loss_train,Loss_test,num_steps = logistic_regression(X_train,Y_train,1e-8,X_test,Y_test)

    print('the coefficents are',w,'and the number of iterations are',num_steps)

    plt.plot(range(1,num_steps+1),Loss_train,'-or',range(1,num_steps+1),Loss_test,'-ob')
    plt.ylabel('Loss function')
    plt.xlabel('# iterations')
    plt.legend(['training loss','test loss'])
    plt.show()


if __name__ == '__main__':
    main()
