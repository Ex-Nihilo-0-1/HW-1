# HW 1

1. Did you alter the Node data structure? If so, how and why?

We did alter the Node data structure. We added `self.decision_label` in the node's init function and added two helper function: `add_label` and `add_decision_label`.

We have these two function so that we can use the add_label function for all the leaf node in the decision tree, and `add_decision_label` for nodes that are not terminal and are used to determine which attribute will be used for that node.

2. How did you handle missing attributes, and why did you choose this strategy?

We handle missing attribute by either skipping it upon detection or create substitute attributes using the most common class label to fill them. 

We use this strategy because it's a simple solution that provided the best stability.

