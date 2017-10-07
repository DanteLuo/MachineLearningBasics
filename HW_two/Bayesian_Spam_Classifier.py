import csv
import os
import numpy as np


# loading data and store them in array
dataset = list()
label_buf = list()
token_list = list()

token_path = os.path.expanduser('~/PycharmProjects/ML/HW_two/spam_classification/spam_classification/TOKENS_LIST')
with open(token_path,newline='') as token:
    reader = csv.reader(token, delimiter=' ')
    for row in reader:
        token_list.append(row)

train_path = os.path.expanduser('~/PycharmProjects/ML/HW_two/spam_classification/spam_classification/SPARSE.TRAIN')
with open(train_path, newline='') as train:
    reader = csv.reader(train, delimiter=' ')
    for row in reader:
        label_buf.append(int(row[0]))
label = np.asarray(label_buf,dtype=int)

nd = len(label)
nw = len(token_list)
count_d_w = np.zeros([nd,nw],dtype=int)
with open(train_path, newline='') as train:
    reader = csv.reader(train, delimiter=' ')
    for d_id, row in enumerate(reader):
        current_email = csv.reader(row[2:-1],delimiter=':')
        for rows in current_email:
            w_id = int(rows[0])
            count = int(rows[1])
            count_d_w[d_id][w_id-1] = count


# calculate p(y=1)=p_y[0] and p(y=-1)=p_y[1]
p_y = np.zeros(2)
p_y[0] = len((np.argwhere(label>0)))
p_y[1] = nd - p_y[0]


# calculate p(wj|yi=y)
p_wj_y = np.zeros([2,nw])

p_wj_y_numerator = np.sum(count_d_w[np.argwhere(label>0)][:],axis=0,dtype=float)[0]+1
p_wj_y_denominator = np.sum(p_wj_y_numerator,dtype=float)
p_wj_y[0] = p_wj_y_numerator/p_wj_y_denominator

p_wj_y_numerator = np.sum(count_d_w[np.argwhere(label<0)][:],axis=0)[0]+1
p_wj_y_denominator = np.sum(p_wj_y_numerator)
p_wj_y[1] = p_wj_y_numerator/p_wj_y_denominator


# classify the test dataset
# read the test dataset
label_test_buf = list()
test_path = os.path.expanduser('~/PycharmProjects/ML/HW_two/spam_classification/spam_classification/SPARSE.TEST')
with open(test_path, newline='') as test:
    reader = csv.reader(test, delimiter=' ')
    for row in reader:
        label_test_buf.append(int(row[0]))
label_test = np.asarray(label_test_buf,dtype=int)

nd_test = len(label_test)
count_d_w_test = np.zeros([nd_test,nw],dtype=int)
with open(test_path, newline='') as test:
    reader = csv.reader(test, delimiter=' ')
    for d_id, row in enumerate(reader):
        current_email = csv.reader(row[2:-1],delimiter=':')
        for rows in current_email:
            w_id = int(rows[0])
            count = int(rows[1])
            count_d_w_test[d_id][w_id-1] = count


# calculate p(y=1|d)
prediction = -(np.ones(nd_test))
for d_index,ground_truth in enumerate(label_test):
    p_di_y1 = 0
    p_di_y0 = 0 # y=-1
    num_step = 0
    for word_index in range(nw):
        if count_d_w_test[d_index][word_index]:
            p_di_y1 += p_wj_y[0][word_index]**count_d_w_test[d_index][word_index]
            p_di_y0 += p_wj_y[1][word_index]**count_d_w_test[d_index][word_index]
            num_step += 1

    prediction_single = p_di_y1*p_y[0]/(p_di_y1*p_y[0]+p_di_y0*p_y[1])
    if prediction_single > 0.5:
       prediction[d_index] = 1

    print('Case finished after {} timesteps'.format(num_step))


# calculate training error
training_err = float(np.count_nonzero((prediction-label_test)))/float(nd_test)*100
print('The error rate on the SPARSE.TRAIN is {}%.'.format(training_err))


# pick n most indicative tokens
num_highest = 5
token_indication = np.log(np.divide(p_wj_y[0],p_wj_y[1]))
highest_id = token_indication.argsort()[-nd_test:][:num_highest]
print('The most indicative token are:')
for indicative_id, token_id in enumerate(highest_id):
    print(indicative_id+1,str(token_list[token_id]))
