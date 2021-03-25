import numpy as np
import math
from activation.activationFunction import linear, sigmoid, relu, softmax
from parameter_reader import read_parameter
from csv_reader import read_model, read_data
import random

NEURON_INPUT = 4
# For iterating per item in array. Except for linear function because single parameter is automate to iterate per item
linear = np.vectorize(linear)
sigmoid = np.vectorize(sigmoid)
relu = np.vectorize(relu)


class NeuralNetwork:
    def __init__(self):
        self.base_layer = []
        self.current_layer = []
        self.threshold = 0.001
        self.maximum_iter = 100

    def get_total_layer(self):
        return len(self.layer)

    def enqueue_layer(self, layer):
        self.base_layer.append(layer)

    def deque_layer(self):
        self.base_layer.pop(0)

    def forward_propagation(self):
        self.current_layer = self.base_layer.copy()
        for idx in range(len(self.current_layer)):
            print("")
            print("LAYER === " + str(idx))
            if idx != 0:
                self.current_layer[idx].input_value = self.current_layer[idx-1].result
            else:
                print(type(self.current_layer[idx].input_value))
                print("Input layer \t:", self.current_layer[idx].input_value)
            self.current_layer[idx].compute()

    def draw(self):
        from graphviz import Digraph
        f = Digraph('Feed Forward Neural Network', filename='ann1.gv')
        f.attr('node', shape='circle', fixedsize='true', width='0.9')

        for i in range(len(self.current_layer)):
            if i != 0:
                if i == 1:
                    print(self.current_layer[i].weight)
                    print(self.current_layer[i].bias)
                    for j in range(len(self.current_layer[i].weight)):
                        # f.edge("x{i}","hidden{i}")
                        for k in range(len(self.current_layer[i].weight[j])):
                            print(f'x{j}')
                            print(self.current_layer[i].weight[j][k])
                            print(f'h{i}_{k}')
                            f.edge(f'x{j}', f'h{i}_{k}', str(
                                self.current_layer[i].weight[j][k]))
                    for j in range(len(self.current_layer[i].bias)):
                        print(f'bx')
                        print(self.current_layer[i].bias[j])
                        print(f'h{i}_{j}')
                        f.edge(f'bx', f'h{i}_{j}', str(
                            self.current_layer[i].bias[j]))
                else:
                    print(self.current_layer[i].weight)
                    for j in range(len(self.current_layer[i].weight)):
                        for k in range(len(self.current_layer[i].weight[j])):
                            print(f'h{i-1}_{j}')
                            print(self.current_layer[i].weight[j][k])
                            print(f'h{i}_{k}')
                            f.edge(f'h{i-1}_{j}', f'h{i}_{k}',
                                str(self.current_layer[i].weight[j][k]))
                    for j in range(len(self.current_layer[i].bias)):
                        print(f'bhx{i-1}')
                        print(self.current_layer[i].bias[j])
                        print(f'h{i}_{j}')
                        f.edge(f'bhx{i-1}', f'h{i}_{j}',
                            str(self.current_layer[i].bias[j]))

        f.view()
    

    def back_propagation(self):
        for i in range(len(self.current_layer)):
            if i != len(self.current_layer) - 1 and i != 0: #Not input or output layer
                self.current_layer[i].weight = self.current_layer[i].update_weight()
                self.current_layer[i].bias = self.current_layer[i].update_bias()
            elif i != 0:
                self.current_layer[i].weight = self.current_layer[i].update_weight_output()

    def learn(self):
        error = 0 # placeholder
        current_iter = 0
        while error > self.threshold and current_iter < self.maximum_iter:
            self.forward_propagation()
            self.back_propagation()
        
class InputLayer:
    def __init__(self, arr=[]):
        self.input_value = np.array(arr)
        self.result = self.input_value

    def compute(self):
        pass

from chainRule import *
class Layer(InputLayer):
    def __init__(self, neuron_input, neuron_output, activation_function, **kwargs):
        super().__init__([])
        self.weight = np.array([[random.random() for x in range(neuron_output)] for j in range(neuron_input)])
        self.bias = np.array([random.random() for x in range(neuron_output)])
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
        else:
            self.result = np.matmul(self.input_value, self.weight) + self.bias
        print("Sigma \t: ", self.result)

    def compute(self):
        print("Input \t: ", self.input_value)
        self.sigma()
        self.activate()
        print("Weight \t: ", self.weight)
        print("Result \t: ", self.result)
    
    def update_weight(self, arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i):
        pass
    
    def update_weight_output(self, arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i):
        pass

    def update_bias(self, target, out_h, out_o):
        pass
    
class OutputLayer(Layer):
    def __init__(self, neuron, activation_function, **kwargs):
        super().__init__(neuron, activation_function, **kwargs)
        self.error([])

# driver test


def main():
    print('Feed Forward Neural Network : XOR')
    print("=================================")

    parameter = read_parameter()
    data = read_data()
    model = read_model()
    neural_network = NeuralNetwork()
    result = []
    layer = []
    target = []
    print("Activation Layer: ")
    for index, item in model.iterrows():
        act = None
        if (item['activation'] == 'sigmoid'):
            act = sigmoid
        elif (item['activation'] == 'linear'):
            act = linear
        elif (item['activation'] == 'relu'):
            act = relu
        elif (item['activation'] == 'softmax'):
            act = softmax
        if index == 0: layer.append(Layer(NEURON_INPUT, item['neuron'], act, threshold=0.1))
        elif index > 0: layer.append(Layer(model.iloc[index - 1, 0], item['neuron'], act, threshold=0.1))
    print("")
    for index, item in data.iterrows():
        layer.insert(0, InputLayer([item['sepal_length'], item['sepal_width'], item['petal_length'], item['petal_width']]))
        target.append(item['species'])
        neural_network.base_layer = layer
        neural_network.forward_propagation()
        result.append(neural_network.current_layer[-1].result)
        neural_network.deque_layer()
    print("Target Class \t: ", target)
    print("Predict Class \t: ", result)
    print("=================================")
    # if (result == target):
    #     print("Result : Good Predict")
    # else:
    #     print("Result : Wrong Predict")
    
    neural_network.draw()

if __name__ == "__main__":
    main()
    # print(data)
    # print(target)
    # readWeight()
