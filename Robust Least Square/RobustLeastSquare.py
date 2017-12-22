import numpy as np
import matplotlib.pyplot as plt
from csv import reader


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

def weight_func(R):
    return 1/np.sqrt(1+R**2)

def residual_weight(X,Y,W):
    R = Y-np.dot(X,W)
    Q = weight_func(R)
    return np.diag(np.reshape(Q,[len(Q),]))

def IRLS(X, Y, epsilon = 1e-6):
    delta = epsilon + 1.
    W_curr = np.dot(np.linalg.pinv(X),Y)
    W_LS = W_curr

    while delta > epsilon:
        Q = residual_weight(X,Y,W_curr)
        inv_buf = np.linalg.inv(np.dot(np.dot(np.transpose(X),Q),X))
        W_next = np.dot(np.dot(np.dot(inv_buf,np.transpose(X)),Q),Y)
        delta = np.linalg.norm((abs(W_next-W_curr)))
        W_curr = W_next

    return W_curr, W_LS

# loading data
filename = "synthetic.csv"
data = load_csv(filename)

# convert string attributes to integers
for i in range(0, len(data[0])):
    str_column_to_float(data, i)

data = np.asarray(data)
X = np.reshape(data[:,0],[len(data),1])
X = np.append(np.ones([len(X),1]),X,axis=1)
Y = np.reshape(data[:,1],[len(data),1])

W_IRLS, W_LS = IRLS(X,Y)
print('The IRLS results is',W_IRLS,'The least square results is',W_LS)

W_True = np.array([[5],[10]])
Y_IRLS = np.dot(X,W_IRLS)
Y_LS = np.dot(X,W_LS)
Y_True = np.dot(X,W_True)

plt.plot(X[:,1],Y_IRLS,':')
plt.plot(X[:,1],Y_LS,'-.')
plt.plot(X[:,1],Y_True,'--')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(['IRLS','Least Square','True Value'])
plt.show()