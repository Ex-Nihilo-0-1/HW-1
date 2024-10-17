import ID3
import random

class randomForest:
    #initiliaze the tree and max numbers
    def __init__(self, tree_numbers = 10, max_feature = None ):
        self.tree_numbers = tree_numbers
        self.trees = []
        self.max_feature = max_feature


    #random samples
    def randomSample(self, randomExamples, attributes):
        randomExamples = random.choices(randomExamples, k=len(randomExamples))
        if self.max_feature:
            sampledAttribute = random.sample(attributes, self.max_feature)
        else:
            sampledAttribute = attributes
        new = []
        #build tree with attributes
        for example in randomExamples:
            newExample = {}
            for attribute in sampledAttribute:
                value = example.get(attribute, 0)
                newExample[attribute] = value

            newExample['Class'] = example['Class']
            new.append(newExample)

        return new
    # train with decision tree
    def train(self, randomExamples, default):
        firstExample = randomExamples[0]
        keys = list(firstExample.keys())
        attributes = keys[:-1]
        #train the tree with different samples
        for i in range(self.tree_numbers):
            print(f"Training: {i + 1}")
            sample = self.randomSample(randomExamples, attributes) # get the random set
            if not self.max_feature:
                sampleAttributes = attributes
            else:
                sampleAttributes = random.sample(attributes, self.max_feature)

            trained = self.randomSample(sample, sampleAttributes)
            #call ID3 to train
            tree = ID3.ID3(trained, default)
            self.trees.append(tree)

    #predict using the evaluate method in ID3
    def predict(self,randomExamples):
            predictions = []
            #get the all the predictions
            for tree in self.trees:
                print(tree)
                prediction = ID3.evaluate(tree, randomExamples)
                predictions.append(prediction)
                
            counts = {}
            #find the most common prediction
            for prediction in predictions:
                counts[prediction] = counts.get(prediction,0) + 1
            mostClass = max(counts, key = counts.get)

            print(f"Prediction complete: {predictions}")

            return mostClass
    
    #predict method to predict all the test set
    def predictAll(self, test_set):
        predictions = []
        for example in test_set:
            print("testing example: ")
            print(example)
            prediction = self.predict(example)
            predictions.append(prediction)
        return predictions
    
def evaluate(predictions, actual):
    correct = sum([1 for pred, act in zip(predictions, actual) if pred == act])
    return correct / len(actual)

def read_candy_data(file_path):
    with open(file_path, 'r') as f:
        # Read the header
        headers = f.readline().strip().split(',')

        # Read the rest of the lines as data
        data = []
        for line in f:
            values = line.strip().split(',')
            # Create a dictionary for each row with headers as keys
            example = {headers[i]: int(values[i]) for i in range(len(headers))}
            data.append(example)

    return data
candy_data = read_candy_data('candy.data')
print(candy_data[0])
split_index = int(0.8 * len(candy_data))
train_data = candy_data[:split_index]
test_data = candy_data[split_index:]

# Initialize and train the Random Forest
forest = randomForest(tree_numbers=10, max_feature=5)
forest.train(train_data, default=0)
# Get predictions for the test set
forest_predictions = forest.predictAll(test_data)
print(forest_predictions)
# Extract the actual labels from the test data
actual_labels = [example['Class'] for example in test_data]

# Evaluate ID3 single tree accuracy
id3_tree = ID3.ID3(train_data, 0)
id3_accuracy = ID3.test(id3_tree, test_data)
print(f"ID3 Decision Tree Accuracy: {id3_accuracy * 100:.2f}%")
# Evaluate Random Forest accuracy
forest_accuracy = evaluate(forest_predictions, actual_labels)

print(f"ID3 Decision Tree Accuracy: {id3_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {forest_accuracy * 100:.2f}%")