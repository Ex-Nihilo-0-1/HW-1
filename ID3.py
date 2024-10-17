from node import Node
import math
import numpy as np
import parse
import copy

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  #HELPER FUNCTIONS SECTION
  def h(prob):
    '''
    Helper function, computes entropy.
    '''
    if prob == 0 or prob == 1:
        return 0
    return -1*(prob)*np.log2(prob)
  
  def get_da(attribute:str, value:str, examples):
    '''
    Helper function, computes D_a.
    Takes in an attribute and a value,
    returns a list that contains the examples that 
    have the given value for the given attribute.
    '''
    examples = examples
    d_a = []
    for e in examples:
      if attribute in e.keys():
        if e[attribute] == value:
          d_a += [e]
    return d_a
  
  def info_gain(node, attribute:str):
    '''
    Helper function, computes info gain. 
    '''
    
    # return gain
    total_count = len(examples)
    if total_count == 0:
        return 0

    # Calculate entropy of the parent node
    label_counts = {}
    for e in examples:
        label = e['Class']
        label_counts[label] = label_counts.get(label, 0) + 1
    h_node = 0
    for count in label_counts.values():
        prob = count / total_count
        h_node += h(prob)

    # Calculate entropy of the children nodes
    value_counts = {}
    value_label_counts = {}
    for e in examples:
        value = e[attribute]
        label = e['Class']
        value_counts[value] = value_counts.get(value, 0) + 1
        if value not in value_label_counts:
            value_label_counts[value] = {}
        value_label_counts[value][label] = value_label_counts[value].get(label, 0) + 1

    h_children = 0
    for value in value_counts:
        weight = value_counts[value] / total_count
        h_value = 0
        for count in value_label_counts[value].values():
            prob = count / value_counts[value]
            h_value += h(prob)
        h_children += weight * h_value

    gain = h_node - h_children
    return gain
  
  def find_best_split(node, attributes):
    max_gain = info_gain(t, attributes[0])
    a_star = attributes[0]
    for a in attributes:
      gain = info_gain(t, a)
      if gain > max_gain:
        max_gain = info_gain(t, a)
        a_star = a
    return a_star
  #END OF HELPER FUNCTIONS SECTION
  
  if not examples:
          leaf = Node()
          leaf.add_label(default)
          return leaf
  
  # Create a Root node for the tree
  attributes = list(examples[0].keys())[:-1]
  t = Node()
  class_values = []
  for e in examples: 
    class_values += [e['Class']]
  most_common_class_value = max(set(class_values), key = class_values.count)  #finds the mode of Class values
  t.add_label(most_common_class_value)

  # If all Examples are LABEL, Return the single-node tree Root, with positive label= LABEL 
  if len(set(class_values)) == 1: #if set(class_values) is homogeneous return t
    return t
 
  # If Attributes is empty, Return the single-node tree Root, with label = most common value of Target_attribute in Examples
  if attributes == []:
    return t
  
  #find_best_split 

  a_star = find_best_split(t, attributes)     #Find A*, the best label to split this node on
  t.add_decision_label(a_star)                #Set the decision label of the current node to that
  a_star_values = []                          #Initialize variable for all possible values of label A*
  for e in examples:                        
    a_star_values += [e[a_star]]              
  a_star_values = list(set(a_star_values))    #Compute all possible values of label A*

  for a in a_star_values:                     #Compute the subset D_a, the items in the training set such that has value a for label A*
    d_a = get_da(a_star, a, examples)         
    if d_a == []:                             #If D_a is empty, then ...
      child = Node()
      child.add_label(most_common_class_value)
      t.children[a] = child
    else:                                     #If D_a is non-empty, then ...
      d_a_copy = []
      for e in d_a:
        e_copy = e.copy()
        del e_copy[a_star]
        if 'Class' in e_copy:
          d_a_copy.append(e_copy)
      t.children[a] = ID3(d_a_copy, default)

  return t

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  if node.children == {}:
     return
  threshold = 5
  decision_label = node.decision_label
  values = list(node.children.keys())
  for value in values:
    d_a = []
    for e in examples:
      if decision_label in e.keys():
        if e[decision_label] == value:
          d_a += [e]
    if len(d_a) <= threshold:
       for e in examples:
          if value in e.keys():
            del e[value]
       del node.children[value]
  
  for child in node.children.values():
     prune(child, examples)


def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''

  correct = 0
  total = len(examples)
  for example in examples:
      prediction = evaluate(node, example)
      if prediction == example['Class']:
          correct += 1
  return correct / total if total > 0 else 0


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

  tree = node
  if tree.decision_label == None:
    return tree.label
  else:
    example_value = example[tree.decision_label]
    if example_value in tree.children:
        return evaluate(tree.children[example_value], example)
    else:
        # Return the current node's label if the child doesn't exist
        return tree.label
