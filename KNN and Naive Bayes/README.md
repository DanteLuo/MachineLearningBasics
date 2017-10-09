# Naive Bayes and KNN classification

## Naive Bayes

### What is it?
It is a very simple but powerful classifier.
[Machine Leaning Bayes Classifier](http://cs229.stanford.edu/notes/cs229-notes2.pdf)
is a great document explains everything.

It is a generative method.

### What's in the package?
It is an implementation of Naive Bayes Classifier for picking up the 
spam. The data is stored in the spam-classification folder, which 
includes vocabulary, test and training data.

### How does this py file work?
Run the Bayesian_Spam_Classifier and it will give you training error
and the most important or indicative words.


## K Nearest Neighbors

### What is it?
[Kevin's Blog](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/#writing-our-own-knn-from-scratch)

It is a classification method comparing the distance or the norm
between the data we gonna predict and the training data we have.

Get the K, which is a non-negative integer number, closet points
according to norm calculated and make prediction based on these 
points.

### What's in the package?
It is an implementation of K-NN with [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
It runs through several K and generates a plot of the training error.
The error is averaged to avoid the noise.

### How does this py file work?
Run the KNN.py then you will see the training error plot and 
please change some of the parameters like K and see how does it affect
the decision.

PS: Please install the requirements.txt before run the classifier.
For how to install requirements file please check [here](https://pip.pypa.io/en/latest/user_guide/#requirements-files).
