import ID3
import random

class randomForest:
    def __init__(self, tree_numbers = 10, max_feature = None ):
        self.tree_numbers = tree_numbers
        self.trees = []
        self.max_feature = max_feature


    
    def randomSample(self, randomExamples, attributes):
        new = []
        for example in randomExamples:
            new_samples = {}
            for i in attributes:
                new_samples[i] = example[i]
            new_samples['Class'] = example['Class']
            new.append(new_samples)

        return new
    
    def train(self, randomExamples, default):
        firstExample = randomExamples[0]
        keys = list(firstExample.keys())
        attributes = keys[:-1]

        for i in range(self.tree_numbers):
            print(f"Training: {i + 1}")
            sample = self.randomSample(randomExamples, attributes) # get the random set
            if not self.max_feature:
                sampleAttributes = attributes
            else:
                sampleAttributes = random.sample(attributes, self.max_feature)

            trained = self.randomSample(sample, sampleAttributes)
            tree = ID3.ID3(trained, default)
            self.trees.append(tree)
