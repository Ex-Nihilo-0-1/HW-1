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
    possible_values = []
    label_count = 0                   
    total_count = len(examples)         
    for e in examples:  
      possible_values += [e[attribute]]
      if e['Class'] == node.label:
        label_count += 1
    keys = set(possible_values)

    count_dict = dict.fromkeys(keys, 0)
    for v in list(set(possible_values)):
      for e in examples:
        if e[attribute] == v:
          count_dict[v] += 1
    
    h_node = h(label_count/total_count) + h(1-(label_count/total_count))
    h_children = 0
    for k in list(count_dict.keys()):
      h_children += h(count_dict[k]/total_count)
    gain = h_node - h_children

    return gain
  #END OF HELPER FUNCTIONS SECTION

  # Create a Root node for the tree
  attributes = list(examples[0].keys())[:-1]
  t = Node()

  class_values = []
  for e in examples: 
    class_values += [e['Class']]
  max_label = max(set(class_values), key = class_values.count)  #finds the mode of Class values
  t.add_label(max_label)

  # If all Examples are positive, Return the single-node tree Root, with label= - 
  # If all Examples are negative, Return the single-node tree Root, with label =-
  if set(class_values) == {'1'}:
    return t
  if set(class_values) == {'0'}:
    return t
  
  # If Attributes is empty, Return the single-node tree Root, with label = most common value of Target_attribute in Examples
  if attributes == []:
    return t
  

  max_gain = info_gain(t, attributes[0])
  a_star = attributes[0]
  for a in attributes:
    ig = info_gain(t, a)
    if ig > max_gain:
      max_gain = info_gain(t, a)
      a_star = a
  t.add_decision_label(a_star)
  a_star_values = []
  for e in examples:
    a_star_values += [e[a_star]]
  a_star_values = list(set(a_star_values))
  for a in a_star_values:
    d_a = get_da(a_star, a, examples)
    if d_a == []:
      child = Node()
      child.add_label(max_label)
      t.children[a] = child
    else:
      for e in d_a:
        del e[a_star]
      t.children[a] = ID3(d_a, default)

  return t

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''


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
