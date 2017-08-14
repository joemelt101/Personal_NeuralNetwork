from nn import neuralnetwork
from unittest import TestCase
import numpy

class TestNeuralNetwork(TestCase):

    def test_createNetworkCalled_ReturnsTupleWithCorrectDimensions(self):
        n = neuralnetwork.create_network(1)
        self.assertEqual(len(n), 3)

    def test_createNetworkCalled_ReturnsNetworkWithProperAlphaDimensions(self):
        expectedDimensions = [3, 4, 5]

        (w, a, b) = neuralnetwork.create_network(expectedDimensions[0],
                                         expectedDimensions[1],
                                         expectedDimensions[2])

        for l in range(1, len(expectedDimensions)):
            self.assertEqual(len(a[l]), expectedDimensions[l])

    def test_createNetworkCalled_ReturnsNetworkWithProperBetaDimensions(self):
        expectedDimensions = [3, 4, 5]

        (w, a, b) = neuralnetwork.create_network(expectedDimensions[0],
                                         expectedDimensions[1],
                                         expectedDimensions[2])

        for l in range(1, len(expectedDimensions)):
            self.assertEqual(len(b[l]), expectedDimensions[l])

    def test_createNetworkCalled_ReturnsNetworkWithProperWeightsHeightDimensions(self):
        expectedDimensions = [3, 4, 5]

        (w, a, b) = neuralnetwork.create_network(expectedDimensions[0],
                                         expectedDimensions[1],
                                         expectedDimensions[2])

        # Assert

        # We don't care about the first layer of weights because they are not applied

        for l in range(1, len(expectedDimensions)):
            expectedHeight = expectedDimensions[l]
            self.assertEqual(len(w[l]), expectedHeight)

    def test_createNetworkCalled_ReturnsNetworkWithProperWeightsWidthDimensions(self):
        expectedDimensions = [3, 4, 5]

        (w, a, b) = neuralnetwork.create_network(expectedDimensions[0],
                                         expectedDimensions[1],
                                         expectedDimensions[2])

        # Assert

        # We don't care about the first layer of weights because they are not applied

        for l in range(1, len(expectedDimensions)):
            expectedWidth = expectedDimensions[l - 1]
            for j in range(1, len(w[l])):
                self.assertEqual(len(w[l][j]), expectedWidth)
            

    #     self.assertEqual(L, 3)
    # def test_createNetworkCalled_ReturnsNetworkWithProperOutputDimensions(self):
    #     #Arrange, act, assert
    #     expected = [1, 2, 3]

    #     n = neuralnetwork.create_network(1, 2, 3)
        
    #     for i in range(0, len(expected)):
    #         actualAlphaLength = n[i]
    #         self.assertEqual(len(n[i]), )

