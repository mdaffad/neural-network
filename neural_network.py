import numpy as np
import math
from activation.activationFunction import linear, sigmoid, relu, softmax
from dataReader import *

# For iterating per item in array. Except for linear function because single parameter is automate to iterate per item
linear = np.vectorize(linear)
sigmoid = np.vectorize(sigmoid)
relu = np.vectorize(relu)


class NeuralNetwork:
    def __init__(self):
        self.base_layer = []
        self.current_layer = []

    def get_total_layer(self):
        return len(self.layer)

    def enqueue_layer(self, layer):
        self.base_layer.append(layer)

    def deque_layer(self):
        self.base_layer.pop(0)

    def solve(self):
        self.current_layer = self.base_layer.copy()
        for idx in range(len(self.current_layer)):
            if idx != 0:
                self.current_layer[idx].input_value = self.current_layer[idx-1].result
            self.current_layer[idx].compute()


class InputLayer:
    def __init__(self, arr=[]):
        self.input_value = np.array(arr)
        self.result = self.input_value

    def compute(self):
        pass


class Layer(InputLayer):
    def __init__(self, arr_weight, arr_bias, activation_function, **kwargs):
        super().__init__([])
        self.weight = np.array(arr_weight)
        self.bias = np.array(arr_bias)
        self.result = np.array([])
        self.activation_function = activation_function
        self.kwargs = kwargs

    def activate(self):
        self.result = self.activation_function(self.result, self.kwargs)

    def sigma(self):
        # case 1 Dimension
        if(len(self.weight) == 1):
            self.result = np.matmul(
                self.input_value, self.weight.flatten()) + self.bias
        else:
            self.result = np.matmul(np.transpose(
                self.input_value), self.weight) + self.bias

    def compute(self):
        self.sigma()
        self.activate()

# driver test


def main():
    print('Feed Forward Neural Network : XOR')
    print("=================================")

    data_training, target = readData()
    activation, bias, weight = readWeight('model 1.txt')
    neural_network = NeuralNetwork()
    result = []
    layer = []

    print("Activation \t: ", end="")
    for i in range(len(activation)):
        act = None
        if (activation[i] == 'sigmoid'):
            act = sigmoid
        elif (activation[i] == 'linear'):
            act = linear
        elif (activation[i] == 'relu'):
            act = relu
        elif (activation[i] == 'softmax'):
            act = softmax
        print(activation[i], end=" ")
        layer.append(Layer(weight[i], bias[i], act, threshold=0.1))
    print("")
    # layer.append(Layer([[20, -20], [20, -20]], [-10, 30], sigmoid, threshold=0.1))
    # layer.append(Layer([[20, 20]], [-30], sigmoid,  threshold=0.1))
    for data in data_training:
        layer.insert(0, InputLayer(data))
        neural_network.base_layer = layer
        neural_network.solve()
        result.append(neural_network.current_layer[-1].result)
        neural_network.deque_layer()
    print("Target Class \t: ", target)
    print("Predict Class \t: ", result)
    print("=================================")
    if (result == target):
        print("Result : Good Predict")
    else:
        print("Result : Wrong Predict")


if __name__ == "__main__":
    main()
    # print(data)
    # print(target)
    # readWeight()
