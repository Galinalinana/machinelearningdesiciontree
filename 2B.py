#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:06:13 2019

@author: nana


#I, Bohan Gou, 000360941 certify that this material is my original work.
# No other person's work has been used without due acknowledgement.
# I have not made my work available to anyone else

"""
 
from sklearn import datasets
from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np 
from sklearn.datasets import load_wine

#with open ("data.csv") as file:
#    reader= csv.DictReader(file, delimiter=",")
#    read=csv.reader(delimiter=",")
read=load_wine()
    
indices = np.random.permutation(len(read.data))  # permutes the numbers from 0 to len-1
    
split = round(len(indices) * 0.8)                   # Number of training data items
    
train = read.data[indices[:split]]               # Now get the training/testing sets
train_target = read.target[indices[:split]]
test = read.data[indices[split:]]
test_target = read.target[indices[split:]]
         
clf = tree.DecisionTreeClassifier()
#        clf.fit(file.data[:,:-1], file.target)
clf.fit(read.data[:,:4],read.target) 
    
tree.export.export_graphviz(clf, out_file="tree1.dot",   class_names=read.target_names)
#     

from sklearn import tree

clf = tree.DecisionTreeClassifier() 
clf = clf.fit(train, train_target) 

print ("============ Training Date======================")
prediction = clf.predict(train)
marking = (prediction ==train_target)
correct = len(np.argwhere(marking==True))
print ("Correct: ",correct)
print ("Incorrect: ", len(prediction)- correct)
print ("Accuracy: ", correct/len(prediction) *100)
 
print ("============ Testing Date======================")
prediction = clf.predict(test)
marking = (prediction ==test_target)
correct = len(np.argwhere(marking==True))
print ("Correct: ",correct)
print ("Incorrect: ", len(prediction)- correct)
print ("Accuracy: ", correct/len(prediction) *100)

#====================================================
#
#read=load_wine()
#    
#indices = np.random.permutation(len(read.data))  # permutes the numbers from 0 to len-1
#    
#split = round(len(indices) * 1.0)                   # Number of training data items
#    
#train = read.data[indices[:split]]               # Now get the training/testing sets
#train_target = read.target[indices[:split]]
#test = read.data[indices[split:]]
#test_target = read.target[indices[split:]]
#         
#clf = tree.DecisionTreeClassifier()
##        clf.fit(file.data[:,:-1], file.target)
#clf.fit(read.data[:,:6],read.target) 
    
tree.export.export_graphviz(clf, out_file="tree2.dot", class_names=read.target_names,max_depth=5)
#     

from sklearn import tree

clf = tree.DecisionTreeClassifier() 
clf = clf.fit(test, test_target) 

print ("============ Training Date======================")
prediction = clf.predict(train)
marking = (prediction ==train_target)
correct = len(np.argwhere(marking==True))
print ("Correct: ",correct)
print ("Incorrect: ", len(prediction)- correct)
print ("Accuracy: ", correct/len(prediction) *100)
 
print ("============ Testing Date======================")
prediction = clf.predict(test)
marking = (prediction ==test_target)
correct = len(np.argwhere(marking==True))
print ("Correct: ",correct)
print ("Incorrect: ", len(prediction)- correct)
print ("Accuracy: ", correct/len(prediction) *100)

#====================================================


read=load_wine()
    
indices = np.random.permutation(len(read.data))  # permutes the numbers from 0 to len-1
    
split = round(len(indices) * 0.5)                   # Number of training data items
    
train = read.data[indices[:split]]               # Now get the training/testing sets
train_target = read.target[indices[:split]]
test = read.data[indices[split:]]
test_target = read.target[indices[split:]]
         
clf = tree.DecisionTreeClassifier()
#        clf.fit(file.data[:,:-1], file.target)
clf.fit(read.data[:,:8],read.target) 
    
tree.export.export_graphviz(clf, out_file="tree3.dot",class_names=read.target_names,max_depth=5)
#     


from sklearn import tree

clf = tree.DecisionTreeClassifier() 
clf = clf.fit(train, train_target) 

print ("============ Training Date======================")
prediction = clf.predict(train)
marking = (prediction ==train_target)
correct = len(np.argwhere(marking==True))
print ("Correct: ",correct)
print ("Incorrect: ", len(prediction)- correct)
print ("Accuracy: ", correct/len(prediction) *100)
 
print ("============ Testing Date======================")
prediction = clf.predict(test)
marking = (prediction ==test_target)
correct = len(np.argwhere(marking==True))
print ("Correct: ",correct)
print ("Incorrect: ", len(prediction)- correct)
print ("Accuracy: ", correct/len(prediction) *100)

#====================================================


read=load_wine()
    
indices = np.random.permutation(len(read.data))  # permutes the numbers from 0 to len-1
    
split = round(len(indices) * 0.1)                   # Number of training data items
    
train = read.data[indices[:split]]               # Now get the training/testing sets
train_target = read.target[indices[:split]]
test = read.data[indices[split:]]
test_target = read.target[indices[split:]]
         
clf = tree.DecisionTreeClassifier()
#        clf.fit(file.data[:,:-1], file.target)
clf.fit(read.data[:,:3],read.target) 
    
tree.export.export_graphviz(clf, out_file="tree4.dot",class_names=read.target_names, max_depth=4)
#     

from sklearn import tree

clf = tree.DecisionTreeClassifier() 
clf = clf.fit(train, train_target) 

print ("============ Training Date======================")
prediction = clf.predict(train)
marking = (prediction ==train_target)
correct = len(np.argwhere(marking==True))
print ("Correct: ",correct)
print ("Incorrect: ", len(prediction)- correct)
print ("Accuracy: ", correct/len(prediction) *100)
 
print ("============ Testing Date======================")
prediction = clf.predict(test)
marking = (prediction ==test_target)
correct = len(np.argwhere(marking==True))
print ("Correct: ",correct)
print ("Incorrect: ", len(prediction)- correct)
print ("Accuracy: ", correct/len(prediction) *100)

#====================================================


read=load_wine()
    
indices = np.random.permutation(len(read.data))  # permutes the numbers from 0 to len-1
    
split = round(len(indices) * 0.01)                   # Number of training data items
    
train = read.data[indices[:split]]               # Now get the training/testing sets
train_target = read.target[indices[:split]]
test = read.data[indices[split:]]
test_target = read.target[indices[split:]]
         
clf = tree.DecisionTreeClassifier()
#        clf.fit(file.data[:,:-1], file.target)
clf.fit(read.data[:,:10],read.target) 
    
tree.export.export_graphviz(clf, out_file="tree5.dot",class_names=read.target_names, max_depth=5)
#     

from sklearn import tree

clf = tree.DecisionTreeClassifier() 
clf = clf.fit(train, train_target) 

print ("============ Training Date======================")
prediction = clf.predict(train)
marking = (prediction ==train_target)
correct = len(np.argwhere(marking==True))
print ("Correct: ",correct)
print ("Incorrect: ", len(prediction)- correct)
print ("Accuracy: ", correct/len(prediction) *100)
 
print ("============ Testing Date======================")
prediction = clf.predict(test)
marking = (prediction ==test_target)
correct = len(np.argwhere(marking==True))
print ("Correct: ",correct)
print ("Incorrect: ", len(prediction)- correct)
print ("Accuracy: ", correct/len(prediction) *100)









