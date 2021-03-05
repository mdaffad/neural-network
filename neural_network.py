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
            print("")
            print("LAYER === " + str(idx))
            if idx != 0:
                self.current_layer[idx].input_value = self.current_layer[idx-1].result
            else:
                print("Input layer \t:", self.current_layer[idx].input_value)
                # print(self.current_layer[idx].input_value)
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
        if(len(self.weight[0]) == 1):
            self.result = np.matmul(
                self.input_value, self.weight.flatten()) + self.bias
            # print("masuk 1")

        else:
            self.result = np.matmul(self.input_value, self.weight) + self.bias
        print("Sigma \t: ", self.result)
        # print(self.result)

    def compute(self):
        print("Input \t: ", self.input_value)
        # print(self.input_value)
        self.sigma()
        self.activate()

        print("Weight \t: ", self.weight)
        # print(self.weight)
        print("Result \t: ", self.result)
        # print(self.result)
        # print("transpose")
        # print(np.transpose(
        #     self.input_value))
        # print("weight")

        # print(self.weight)
        # print("result")
        # print(self.result)
# driver test


def main():
    print('Feed Forward Neural Network : XOR')
    print("=================================")

    data_training, target = readData()
    activation, bias, weight = readWeight('model 2.txt')
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
    from graphviz import Digraph
    f = Digraph('Feed Forward Neural Network', filename='ann.gv')
    f.attr('node', shape='circle', fixedsize='true', width='0.9')

    for i in range(len(neural_network.current_layer)):
        if i !=0:
            if i == 1:
                print(neural_network.current_layer[i].weight)
                print(neural_network.current_layer[i].bias)
                for j in range(len(neural_network.current_layer[i].weight)):
                    # f.edge("x{i}","hidden{i}")
                    for k in range(len(neural_network.current_layer[i].weight[j])):
                        print(f'x{j}')
                        print(neural_network.current_layer[i].weight[j][k])
                        print(f'h{i}_{k}')
                        f.edge(f'x{j}', f'h{i}_{k}', str(neural_network.current_layer[i].weight[j][k]))
                for j in range(len(neural_network.current_layer[i].bias)):
                    print(f'bx')
                    print(neural_network.current_layer[i].bias[j])
                    print(f'h{i}_{j}')
                    f.edge(f'bx', f'h{i}_{j}', str(neural_network.current_layer[i].bias[j]))
            else:
                print(neural_network.current_layer[i].weight)
                for j in range(len(neural_network.current_layer[i].weight)):
                    for k in range(len(neural_network.current_layer[i].weight[j])):
                        print(f'h{i-1}_{j}')
                        print(neural_network.current_layer[i].weight[j][k])
                        print(f'h{i}_{k}')
                        f.edge(f'h{i-1}_{j}', f'h{i}_{k}', str(neural_network.current_layer[i].weight[j][k]))
                for j in range(len(neural_network.current_layer[i].bias)):
                    print(f'bhx{i-1}')
                    print(neural_network.current_layer[i].bias[j])
                    print(f'h{i}_{j}')
                    f.edge(f'bhx{i-1}', f'h{i}_{j}', str(neural_network.current_layer[i].bias[j]))

    f.view()

if __name__ == "__main__":
    main()
    # print(data)
    # print(target)
    # readWeight()
