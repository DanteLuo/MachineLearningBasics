from __future__ import print_function
import operator


# hats off for Josh Gordon, this code is a modification from his, with adding a depth control


training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]


header = ["color", "diameter", "label"]


def is_num(value):
    return isinstance(value,int) or isinstance(value,float)


class Question:

    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def match(self,example):
        value = example[self.feature]
        if is_num(value):
            return self.value <= value
        else:
            return self.value == value

    def __repr__(self):
        condition = "=="
        if is_num(self.value):
            condition = ">="
        return "Is %s %s %s?" % (header[self.feature],condition,str(self.value))


def partition(dataset, question):

    true_data, false_data = [], []

    for sample in dataset:
        if question.match(sample):
            true_data.append(sample)
        else:
            false_data.append(sample)

    return true_data, false_data


def class_count(dataset):
    count = {}

    for sample in dataset:
        label = sample[-1]
        if label in count:
            count[label] += 1
        else:
            count[label] =1

    return count

# Gini impurity is our Entropy
def gini(dataset):

    counts_class = class_count(dataset)
    # print(counts_class)
    impurity = 1.

    for label in counts_class.keys():
        prob = counts_class[label]/float(len(dataset))
        impurity -= prob**2

    return impurity

# Caculate the entropy reduction
def info_gain(true_data, false_data, last_entropy):
    p_true = float(len(true_data))/float(len(true_data)+len(false_data))
    return last_entropy - p_true*gini(true_data)-(1-p_true)*gini(false_data)


def find_split(dataset):

    best_gain = 0
    best_question = None
    num_features = len(dataset[0])-1
    dataset_entropy = gini(dataset)

    for feature in range(num_features):

        values = set([sample[feature] for sample in dataset])

        for value in values:
            cur_question = Question(feature,value)
            true_data, false_data = partition(dataset,cur_question)

            if len(true_data) == 0 or len(false_data) == 0:
                continue

            cur_info_gain = info_gain(true_data,false_data,dataset_entropy)

            if cur_info_gain > best_gain:
                best_gain = cur_info_gain
                best_question = cur_question

    return best_gain, best_question


# # linked lists is applied in this tree
# leaves of decision tree
class leaf:
    def __init__(self, dataset):
        self.prediction_sets = class_count(dataset)
        # # or it's better to keep it this way and process the data afterwards
        # total_num = 0
        # for predict_keys in self.prediction_sets.keys():
        #     total_num += self.prediction_sets[predict_keys]
        # for predict_keys in self.prediction_sets.keys():
        #     self.prediction = float(self.prediction_sets)/float(total_num)

# node of decision tree
class decision_node:
    def __init__(self, question, true_node, false_node):
        self.question = question
        self.true_node = true_node
        self.false_node = false_node


def build_tree(dataset, depth):

    gain, question = find_split(dataset)

    if gain == 0 or depth == 0:
        return leaf(dataset)

    # print(depth)
    depth -= 1

    true_data, false_data = partition(dataset,question)

    true_node = build_tree(true_data, depth)

    false_node = build_tree(false_data, depth)

    return decision_node(question,true_node,false_node)


def predicting_sample_helper(tree_node):
    distribution = tree_node.prediction_sets
    total_num = sum(distribution.values())
    probs = {}
    for class_key in distribution.keys():
        probs[class_key] = float(distribution[class_key])/float(total_num)
    return max(probs.iteritems(), key=operator.itemgetter(1))


def predicting_sample(input_sample, tree_node):

    if isinstance(tree_node,leaf):
        return predicting_sample_helper(tree_node)

    if tree_node.question.match(input_sample):
        return predicting_sample(input_sample, tree_node.true_node)
    else:
        return predicting_sample(input_sample, tree_node.false_node)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, leaf):
        print (spacing + "Predict", node.prediction_sets)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_node, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_node, spacing + "  ")


def print_prediction(prediction):
    print('The sample is %s with probability of %d' % (prediction[0][0],prediction[0][1]))
    # print(prediction)

def main():

    my_tree = build_tree(training_data,5)
    print_tree(my_tree)

    prediction = [predicting_sample(training_data[1], my_tree)]
    print_prediction(prediction)


    # # useful test cases
    # test_question =  Question(0,'Green')
    # print(test_question)

    # lots_of_mixing = [['Apple'],
    #                   ['Orange'],
    #                   ['whatever']]
    #
    #
    # # this will return 0
    # print(gini(lots_of_mixing))
    # current_uncertainty = gini(training_data)
    # true_rows, false_rows = partition(training_data, Question(0, 'Green'))
    # print(info_gain(true_rows, false_rows, current_uncertainty))
    # stats = {'a': 1000, 'b': 3000, 'c': 100, 'd': 3000}
    # test = max(stats.iteritems(), key=operator.itemgetter(1))
    # print(test[1])


if __name__ == '__main__':
    main()