class Node:
  def __init__(self):
    self.label = None
    self.children = {}
    self.decision_label = None
  
  def add_label(self, label):
    self.label = label
  def add_decision_label(self, label):
    self.decision_label = label

 
    

  
	# you may want to add additional fields here...