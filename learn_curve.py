import random
import matplotlib.pyplot as plt
import numpy as np
from ID3 import *
from pprint import pprint

def learning_curve(data, train_sizes = list(range(10, 310, 20)), num_runs=100):


    accuracies_with_pruning = []
    accuracies_without_pruning = []

    for size in train_sizes:
        acc_with_pruning = []
        acc_without_pruning = []


        for i in range(num_runs):
            # Shuffle data and split into training and test sets
            random.shuffle(data)
            train_data = data[:size]
            test_data = data[size:]
            # Train tree without pruning
            tree = ID3(train_data, default='democrat')
            acc_without_pruning.append(test(tree, test_data))
            # Train tree with pruning
            pruned_tree = copy.deepcopy(tree)
            prune(pruned_tree, test_data) 
            acc_with_pruning.append(test(pruned_tree, test_data))
            # for example in test_data:
            #     print(evaluate(tree, example))  # Print predictions for the test examples


        # Calculate average accuracies for this training size
        avg_acc_without = np.mean(acc_without_pruning)
        avg_acc_with = np.mean(acc_with_pruning)

        # Store the average accuracies
        accuracies_without_pruning.append(avg_acc_without)
        accuracies_with_pruning.append(avg_acc_with)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, accuracies_with_pruning, label='With Pruning', marker='o')
    plt.plot(train_sizes, accuracies_without_pruning, label='Without Pruning', marker='o')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Accuracy on Test Data')
    plt.title('Learning Curve: Accuracy vs. Training Size')
    plt.legend()
    plt.grid(True)
    plt.savefig("./learning_curve.png")
    plt.close()  
if __name__ == "__main__":
    data = parse.parse("house_votes_84.data")
    learning_curve(data)

