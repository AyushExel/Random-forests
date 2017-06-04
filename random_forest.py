
# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import csv
# Load a CSV file
def load_csv(filename):
	file = open(filename,  "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def cross_validation_split(dataset,n_folds):
    dataset_split = list()
    data_copy = list(dataset)
    fold_size = int(len(dataset)/n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold)< fold_size:
            index = randrange(len(data_copy))
            fold.append(data_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual,predicted):
    score = 0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            score+= 1
    return (score/len(actual))*100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset,n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set,[])
        test_set  =list()
        for row in fold:
            tset = list(row)
            test_set.append(tset)
            tset[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold ]
        accuracy = accuracy_metric(actual,predicted)
        scores.append(accuracy)
    return scores

def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

def gini_index(groups, class_values):
    gini = 0.0
    for value in class_values:
        for group in groups:
            if len(group) == 0:
                continue
            proportion = [row[-1] for row in group].count(value)/float(len(group))
            gini += proportion*(1.0-proportion)
    return gini
        
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset ))
    index,value,group,score = 999,999,None,999
    features = list()
    while len(features)<n_features:
        ranindex = randrange(len(dataset[0])-1)
        if ranindex not in features:
            features.append(ranindex)
    for tindex in features:
        for row in dataset:
            test_group = test_split(tindex,row[tindex],dataset) 
            test_score = gini_index(test_group,class_values)
            if test_score<score:
                index,value,group,score = tindex,row[tindex],test_group,test_score
    return {'index':index,'value':value,'groups':group}


def to_terminal(group):
    classes = [row[-1] for row in group]
    return max(set(classes),key=classes.count)

def split(node, max_depth, min_size, depth,n_features):
    left,right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left']=node['right'] = to_terminal(left+right)
        return
    if depth >= max_depth:
        node['left'],node['right'] = to_terminal(left),to_terminal(right)

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left,n_features)
        split(node['left'],max_depth,min_size,depth+1,n_features)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right,n_features)
        split( node['right'],max_depth,min_size,depth+1,n_features)

def build_tree(train, max_depth, min_size,n_features):
    root = get_split(train,n_features)
    split(root,max_depth,min_size,1,n_features)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'],dict):
            return predict(node['left'],row)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'],row)
        else:
            return node['right']

def subsample(dataset, ratio):
    sample= list()
    length = round(len(dataset)*ratio)
    while len(sample) < length:
        sample.append(dataset[randrange(len(dataset))])
    return sample

def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(dataset,sample_size)
        tree = build_tree(sample,max_depth,min_size,n_features)
        trees.append(tree)
    predictions =[bagging_predict(trees,row) for row in test]
    return predictions

#Test on sample data
'''
seed(1)
filename = 'test_data.csv'

dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm

n_folds = 5
max_depth = 5
min_size = 10
sample_size = 1.0
n_features = 3  #change this according to the feature size (generally used size is sqrt(featureSize))

for n_trees in [1, 5, 10]:
	scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
	print('Trees: %d' % n_trees)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
'''


	