from node import Node
import math
import numpy as np
import parse
import copy
import unit_tests

def ID3(input_examples: list[dict], default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node) 
    trained on the examples. Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''
#---------------------------HELPER FUNCTIONS SECTION---------------------------------
    def h(freq: float) -> float:
        '''
        Helper function, computes entropy.
        '''
        if freq == 0 or freq == 1:
            return 0
        return -1 * freq * np.log2(freq)

    def get_da(attribute: str, value: str, examples: list[dict]) -> list:
        '''
        Helper function, computes D_a.
        Takes in an attribute and a value,
        returns a list that contains the examples that 
        have the given value for the given attribute.
        '''
        d_a = []
        for e in examples:
            if attribute not in e:
                return []
            if e[attribute] == value:
                d_a.append(e)
        return d_a
    
    def possible_values_getter(attribute: str, examples: list[dict]) -> list:
        '''
        given an attribute return a list of all possible values of e[attribute], 
        where e is an entry in examples
        '''
        values = [e[attribute] for e in examples]
        return list(set(values))
    
    def node_classifier(examples):
        '''
        Given data, sort them by their values of 'Class', so that a dictionary like
        {V1: number of entries whose 'Class' value is V1, V2: number of entry whose 'Class' value is V2...} 
        is returned.
        '''
        class_values_count = {}
        for e in examples:
            value = e['Class']
            class_values_count[value] = class_values_count.get(value, 0) + 1

        return class_values_count

    def info_gain(examples: list[dict], attribute: str) -> float:
        '''
        Helper function, computes info gain. 
        '''

        if len(examples) == 0:
            return 0

        # Calculate entropy of the parent node
        class_values_countup = node_classifier(examples)
        
        h_parent = 0

        for count in class_values_countup.values():
            freq = count / len(examples)
            h_parent += h(freq)

        # Calculate entropy of the children nodes
        possible_values = possible_values_getter(attribute, examples)
        h_children = 0
        for v in possible_values:
            d_a = get_da(attribute, v, examples)
            class_values_countup = node_classifier(d_a)
            my_h = 0
            for v in class_values_countup.values():
                freq = v / sum(class_values_countup.values())
                my_h += h(freq)
            h_children += my_h
            

        return h_parent - h_children

    def find_best_split(examples: list[dict], attributes: list[str]) -> str:
        max_gain = info_gain(examples, attributes[0])
        a_star = attributes[0]          #initialize the variable with the first 
        for a in attributes:
            gain = info_gain(examples, a)
            if gain > max_gain:
                max_gain = gain
                a_star = a
        return a_star

#---------------------------END OF HELPER FUNCTIONS SECTION---------------------------------

    if not input_examples:
        leaf = Node()
        leaf.add_label(default)
        return leaf

    # If all examples have the same class label, return a leaf node with that label
    class_values = possible_values_getter('Class', input_examples)
    if len(set(class_values)) == 1:
        leaf = Node()
        leaf.add_label(class_values[0])
        return leaf

    # If no attributes left to split on, return a leaf node with the most common class label
    attributes: list[str] = [attr for attr in input_examples[0].keys() if attr != 'Class']
    if not attributes:
        leaf = Node()
        most_common_class_value = max(set(class_values), key=class_values.count)
        leaf.add_label(most_common_class_value)
        return leaf

    # Find the best attribute to split on
    a_star = find_best_split(input_examples, attributes)
    root = Node()
    root.add_decision_label(a_star)
    root.add_data(input_examples)

    # For each value of the best attribute, create a subtree
    a_star_values = possible_values_getter(a_star, input_examples)
    print(a_star_values)
    for value in a_star_values:
        d_a = get_da(a_star, value, input_examples)
        print("D_a", d_a)
        print("A*", a_star)
        print("A* = ", value)
        if d_a == []:
            child = Node()
            most_common_class_value = max(set(class_values), key=class_values.count)
            child.add_label(most_common_class_value)
            root.children[value] = child
        else:
            for d in d_a:
                del d[a_star]
            child = ID3(d_a, default)
            root.children[value] = child

    return root





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

