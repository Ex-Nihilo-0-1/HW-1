import unittest
import ID3
import parse
import random

class TestID3(unittest.TestCase):
    def test_ID3AndEvaluate(self):
        data = [
            {'a': 1, 'b': 0, 'Class': 1},
            {'a': 1, 'b': 1, 'Class': 1}
        ]
        tree = ID3.ID3(data, 0)
        self.assertIsNotNone(tree, "ID3 test failed -- no tree returned")
        ans = ID3.evaluate(tree, {'a': 1, 'b': 0})
        self.assertEqual(ans, 1, "ID3 test failed.")
        print("ID3 test succeeded.")

    def test_Pruning(self):
        data = [
            {'a': 0, 'b': 1, 'c': 1, 'd': 0, 'Class': 1},
            {'a': 0, 'b': 0, 'c': 1, 'd': 0, 'Class': 0},
            {'a': 0, 'b': 1, 'c': 0, 'd': 0, 'Class': 1},
            {'a': 1, 'b': 0, 'c': 1, 'd': 0, 'Class': 0},
            {'a': 1, 'b': 1, 'c': 0, 'd': 0, 'Class': 0},
            {'a': 1, 'b': 1, 'c': 0, 'd': 1, 'Class': 0},
            {'a': 1, 'b': 1, 'c': 1, 'd': 0, 'Class': 0}
        ]
        validationData = [
            {'a': 0, 'b': 0, 'c': 1, 'd': 0, 'Class': 1},
            {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'Class': 0}
        ]
        tree = ID3.ID3(data, 0)
        ID3.prune(tree, validationData)
        self.assertIsNotNone(tree, "Pruning test failed -- no tree returned.")
        ans = ID3.evaluate(tree, {'a': 0, 'b': 0, 'c': 1, 'd': 0})
        self.assertEqual(ans, 1, "Pruning test failed.")
        print("Pruning test succeeded.")

    def test_ID3AndTest(self):
        trainData = [
            {'a': 1, 'b': 0, 'c': 0, 'Class': 1},
            {'a': 1, 'b': 1, 'c': 0, 'Class': 1},
            {'a': 0, 'b': 0, 'c': 0, 'Class': 0},
            {'a': 0, 'b': 1, 'c': 0, 'Class': 1}
        ]
        testData = [
            {'a': 1, 'b': 0, 'c': 1, 'Class': 1},
            {'a': 1, 'b': 1, 'c': 1, 'Class': 1},
            {'a': 0, 'b': 0, 'c': 1, 'Class': 0},
            {'a': 0, 'b': 1, 'c': 1, 'Class': 0}
        ]
        tree = ID3.ID3(trainData, 0)
        self.assertIsNotNone(tree, "testID3AndTest failed -- no tree returned.")
        acc_train = ID3.test(tree, trainData)
        self.assertEqual(acc_train, 1.0, "Testing on train data failed.")
        acc_test = ID3.test(tree, testData)
        self.assertEqual(acc_test, 0.75, "Testing on test data failed.")
        print("testID3AndTest succeeded.")

    def test_PruningOnHouseData(self):
        inFile = 'house_votes_84.data'
        withPruning = []
        withoutPruning = []
        data = parse.parse(inFile)
        for i in range(100):
            random.shuffle(data)
            train = data[:len(data)//2]
            valid = data[len(data)//2:3*len(data)//4]
            test = data[3*len(data)//4:]
            tree = ID3.ID3(train, 'democrat')
            ID3.prune(tree, valid)
            acc = ID3.test(tree, test)
            withPruning.append(acc)
            tree = ID3.ID3(train + valid, 'democrat')
            acc = ID3.test(tree, test)
            withoutPruning.append(acc)
        avg_with_pruning = sum(withPruning) / len(withPruning)
        avg_without_pruning = sum(withoutPruning) / len(withoutPruning)
        print("Average accuracy with pruning:", avg_with_pruning)
        print("Average accuracy without pruning:", avg_without_pruning)

if __name__ == '__main__':
    unittest.main()
