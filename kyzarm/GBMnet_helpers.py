# GBMnet_helpers.py
# Contains various custom helper functions that aid the main program. 
import numpy

def one_hot(label):
  new_label = numpy.zeros(10)
  new_label[label] = 1
  return new_label

def one_hot_array(labels):
  #Let's make an empty list to store our results.
  new_labels = []
  
  #We'll use a "for" loop to look at each label in the list.
  for label in labels:
    new_labels.append(one_hot(label))

  #Before we're done, we need to turn the list into a numpy array.  We almost always use numpy arrays with neural networks.
  return numpy.asarray(new_labels)