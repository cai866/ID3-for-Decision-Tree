# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import math

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    lableCount={}
    for featValue in x:
        if featValue not in lableCount.keys():
            lableCount[featValue]=0
        lableCount[featValue] +=1
    return lableCount

    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')

def splitDataSet1(dataSet,axis, value):
    retDataSet = []  
    for featVec in dataSet:  
        if featVec[axis] == value:    
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:]) 
            retDataSet.append(reducedFeatVec) 
    return retDataSet


def splitDataSet2(dataSet,axis, value):
    retDataSet = []  
    for featVec in dataSet:  
        if featVec[axis] != value:    
            reducedFeatVec = featVec[:]
            retDataSet.append(reducedFeatVec) 
    return retDataSet

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    numEntries=len(y);
    lableCount=partition(y)

    shannonEnt=0.0
    for key in lableCount:
        pro=float(lableCount[key]/numEntries)
        shannonEnt +=pro*math.log(pro,2)
    return shannonEnt
            
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    
    
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


#find the majority label
def majorityCnt(classList):  
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys(): classCount[vote] = 0  
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def chooseBestFeatureToSplit(dataSet):  
    numFeatures = len(dataSet[0])-1  
    y1=[example[0] for example in dataSet]
    baseEntropy = entropy(y1)
    bestInfoGain = 0.0; bestFeature = -1  
    for i in range(1:numFeatures):  
        featList = [example[i] for example in dataSet]  
        uniqueVals = set(featList)  
        newEntropy = 0.0  
        for value in uniqueVals:  
            subDataSet = splitDataSet1(dataSet, i , value)  
            prob = len(subDataSet)/float(len(dataSet))  
            y2=[example[0] for example in subDataSet]
            newEntropy += prob * entropy(y2)  
            y2=[]
        infoGain = baseEntropy - newEntropy  
        if(infoGain > bestInfoGain):  
            bestInfoGain = infoGain  
            bestFeature = i  

    return bestFeature  

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    if attribute_value_pairs==None:
        for i in range(1,7):
            a = [example[i] for example in x]
            b=[i]
            c=[(x, y) for x in b for y in a]
            attribute_value_pairs.append(c)
            
    classList = y;    
    if classList.count(classList[0]) == len(classList):  
        return classList[0];  
    if (len(attribute_value_pairs) == 0):  
        return majorityCnt(classList);
    if depth >= max_depth:
        return majorityCnt(classList);
                                 
    bestFeat = chooseBestFeatureToSplit(x);   ）
                                  
    featValues = [example[bestFeat] for example in x]  
    uniqueVals = set(featValues)  
    for value in attribute_value_pairs:  
        if(value[0]== bestFeat):                        
            depth +=1
            list=value
            myTree = {(bestFeat,list[1],False):{},(bestFeat,list[1],True):{}}  
            attribute_value_pairs.remove(value)
            x1=splitDataSet1(x, bestFeat, list[1])                         
            x2=splitDataSet2(x, bestFeat, list[1])
            y1=[example[0] for example in x1]
            y2=[example[0] for example in x2]
            myTree[(bestFeat,list[1],False)]= id3(x2,y2,attribute_value_pairs ,depth, max_depth=5)  
            myTree[(bestFeat,list[1],True)]=id3(x1, y1,attribute_value_pairs,depth, max_depth=5)                          
    return myTree 
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    if tree
    firstStr1= tree.keys()[0]
    firstStr2= tree.keys()[1]
    featIndex = tree.keys()[0][0]
    if x[featIndex]==tree.keys()[0][1]:
        if tree[firstStr2]==1 or tree[firstStr2]==0:
            return tree[firstStr2]
        else:
            secondDict = tree[firstStr2]
            classLabel = classify(x,secondDict)
         
    if x[featIndex]!=tree.keys()[0][1]:
        if tree[firstStr1]==1 or tree[firstStr1]==0:
            return tree[firstStr1]
        else:
            secondDict = tree[firstStr1]
            classLabel = classify(x,secondDict)
   
    return classLabel

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    num=len(y_true)
    count=0
    for i in range(num):
        if y_true[i] != y_pred[i]:
            count +=1
    error=count/num
    return error
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')

#compute the confusion matrix and return it
def confusion_matrix(y_true, y_pred)
    num=len(y_true)
    matrix=([0,0],[0,0])
    
    for i in range(num):
        if y_true[i] ==1 and y_pred[i]=1:
            matrix[0][0]+=1
        if y_true[i] ==1 and y_pred[i]=0:
            matrix[0][1]+=1
        if y_true[i] ==0 and y_pred[i]=1:
            matrix[1][0]+=1
        if y_true[i] ==0 and y_pred[i]=0:
            matrix[1][1]+=1
    return matrix

def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))



if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    
     # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    visualize(decision_tree)

    
    # Learn a decision tree of depth i and print the error
    print('Monk1:','\n')
    for i in range(1:11):
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        newtst_err1 = compute_error(ytst, y_pred)
        print('depth ',i,'Test Error = {0:4.2f}%.'.format(newtst_err1 * 100),'   ')
        
    #computer confusion matrix of the tree with depth 1
    decision_tree = id3(Xtrn, ytrn, max_depth=1)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    matrix1 = confusion_matrix(y_true, y_pred)
    print('monk1 confusion matrix trained by the tree with depth 1:',matrix1)

    #computer confusion matrix of the tree with depth 2
    decision_tree = id3(Xtrn, ytrn, max_depth=2)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    matrix2 = confusion_matrix(y_true, y_pred)
    print('monk1 confusion matrix trained by the tree with depth 2:',matrix2)
    
    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))




    #Monk2 testError from depth 1 to 10
    if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    print('\n','Monk2:','\n')
    # Learn a decision tree of depth i and print the error
    for i in range(1:11):
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        newtst_err2 = compute_error(ytst, y_pred)
        print('depth ',i,'Test Error = {0:4.2f}%.'.format(newtst_err2 * 100),'   ')


        
    #Monk3 testError from depth 1 to 10
    if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    rint('\n','Monk3:','\n')
    # Learn a decision tree of depth i and print the error
    for i in range(1:11):
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        newtst_err3 = compute_error(ytst, y_pred)
        print('depth ',i,'Test Error = {0:4.2f}%.'.format(tnewtst_err3 * 100),'   ')


