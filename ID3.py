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
    def h(prob: float) -> float:
        '''
        Helper function, computes entropy.
        '''
        if prob == 0 or prob == 1:
            return 0
        return -1 * prob * np.log2(prob)

    def get_da(attribute: str, value: str, examples: list[dict]) -> list[dict]:
        '''
        Helper function, computes D_a.
        Takes in an attribute and a value,
        returns a list that contains the examples that 
        have the given value for the given attribute.
        '''
        d_a = []
        for e in examples:
            if e[attribute] == value:
                d_a.append(e)
        return d_a

    def info_gain(examples: list[dict], attribute: str) -> float:
        '''
        Helper function, computes info gain. 
        '''
        examples_count = len(examples)
        if examples_count == 0:
            return 0

        # Calculate entropy of the parent node
        class_values_count = {}
        for e in examples:
            value = e['Class']
            class_values_count[value] = class_values_count.get(value, 0) + 1

        h_parent = 0
        for count in class_values_count.values():
            prob = count / examples_count
            h_parent += h(prob)

        # Calculate entropy of the children nodes
        
        attributes_split_by_class = {}
        for e in examples:
            attribute_value = e[attribute]
            class_value = e['Class']
            if attribute_value not in attributes_split_by_class.values():
                attributes_split_by_class[attribute_value] = {}
            attributes_split_by_class[attribute_value][class_value] = attributes_split_by_class[attribute_value].get(class_value, 0) + 1

        h_children = 0
        for key in attributes_split_by_class:
            weight = sum(attributes_split_by_class[key].values())/ examples_count
            h_value = 0
            for count in attributes_split_by_class[key].values():
                prob = count / sum(attributes_split_by_class[key].values())
                h_value += h(prob)
            h_children += weight * h_value

        gain = h_parent - h_children
        return gain

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
    class_values = [e['Class'] for e in input_examples]
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

    # For each value of the best attribute, create a subtree
    a_star_values = set(e[a_star] for e in input_examples)
    for value in a_star_values:
        d_a = get_da(a_star, value, input_examples)
        if not d_a:
            child = Node()
            most_common_class_value = max(set(class_values), key=class_values.count)
            child.add_label(most_common_class_value)
            root.children[value] = child
        else:
            d_a_copy = []
            for e in d_a:
                e_copy = copy.copy(e)
                del e_copy[a_star]
                d_a_copy.append(e_copy)
            child = ID3(d_a_copy, default)
            root.children[value] = child

    return root


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''


  root_node = node  # Keep a reference to the root of the tree

  def prune_node(current_node):
    best_accuracy = test(root_node, examples)
    def get_majority_label(node):
      labels = []
      def collect_labels(current_node):
          if current_node.decision_label is None:
              labels.append(current_node.label)
          else:
              for child in current_node.children.values():
                  collect_labels(child)

      collect_labels(node)
      return max(set(labels), key=labels.count)

    
    if current_node.decision_label is None:
        return  # This is a leaf node

    # Recursively prune child nodes
    for child in current_node.children.values():
        prune_node(child)

    # Attempt to prune this node
    saved_decision_label = current_node.decision_label
    saved_children = current_node.children

    # Make this node a leaf node
    current_node.decision_label = None
    current_node.label = get_majority_label(current_node)  # Set the label to the majority class of the subtree
    current_node.children = {}

    # Evaluate accuracy after pruning
    pruned_accuracy = test(root_node, examples)
    #if pruned_accuracy >= best_accuracy:
        # Keep pruning since accuracy has not decreased
        #best_accuracy = pruned_accuracy
    if pruned_accuracy < best_accuracy:
        # Restore the original node since pruning did not improve accuracy
        current_node.decision_label = saved_decision_label
        current_node.children = saved_children
 
    return 
  
  prune_node(root_node)

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

examples =  parse.parse("cars_train.data")
t = ID3(examples, 0)