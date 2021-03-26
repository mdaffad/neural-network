import numpy as np
import math
from activation.activationFunction import linear, sigmoid, relu, softmax
from parameter_reader import read_parameter
from csv_reader import read_model, read_data
import random
from activation.activationFunction import *
NEURON_INPUT = 4
# For iterating per item in array. Except for linear function because single parameter is automate to iterate per item
linear = np.vectorize(linear)
sigmoid = np.vectorize(sigmoid)
relu = np.vectorize(relu)


class NeuralNetwork:
    def __init__(self, base_layer, learning_rate = 0.001, error_threshold = 0.001, max_iter = 100, batch_size = 1):
        self.base_layer = base_layer
        self.current_layer = base_layer.copy()
        self.learning_rate = learning_rate
        self.error_threshold = error_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size

    def get_total_layer(self):
        return len(self.layer)

    def enqueue_layer(self, layer):
        self.current_layer.insert(0, layer)

    def deque_layer(self):
        self.current_layer.pop(0)

    def forward_propagation(self):
        for idx in range(len(self.current_layer)):
            print("")
            print("LAYER === " + str(idx))
            if idx != 0:
                self.current_layer[idx].input_value = self.current_layer[idx-1].result
            else:
                # print(type(self.current_layer[idx].input_value))
                print("Input layer \t:", self.current_layer[idx].input_value)
            self.current_layer[idx].compute()

    def draw(self):
        from graphviz import Digraph
        f = Digraph('Feed Forward Neural Network', filename='ann1.gv')
        f.attr('node', shape='circle', fixedsize='true', width='0.9')

        for i in range(len(self.current_layer)):
            if i != 0:
                if i == 1:
                    # print(self.current_layer[i].weight)
                    # print(self.current_layer[i].bias)
                    for j in range(len(self.current_layer[i].weight)):
                        # f.edge("x{i}","hidden{i}")
                        for k in range(len(self.current_layer[i].weight[j])):
                            # print(f'x{j}')
                            # print(self.current_layer[i].weight[j][k])
                            # print(f'h{i}_{k}')
                            f.edge(f'x{j}', f'h{i}_{k}', str(
                                self.current_layer[i].weight[j][k]))
                    for j in range(len(self.current_layer[i].bias)):
                        # print(f'bx')
                        # print(self.current_layer[i].bias[j])
                        # print(f'h{i}_{j}')
                        f.edge(f'bx', f'h{i}_{j}', str(
                            self.current_layer[i].bias[j]))
                else:
                    # print(self.current_layer[i].weight)
                    for j in range(len(self.current_layer[i].weight)):
                        for k in range(len(self.current_layer[i].weight[j])):
                            # print(f'h{i-1}_{j}')
                            # print(self.current_layer[i].weight[j][k])
                            # print(f'h{i}_{k}')
                            f.edge(f'h{i-1}_{j}', f'h{i}_{k}',
                                str(self.current_layer[i].weight[j][k]))
                    for j in range(len(self.current_layer[i].bias)):
                        # print(f'bhx{i-1}')
                        # print(self.current_layer[i].bias[j])
                        # print(f'h{i}_{j}')
                        f.edge(f'bhx{i-1}', f'h{i}_{j}',
                            str(self.current_layer[i].bias[j]))

        f.view()

    def learn(self, data):
         # placeholder
        current_iter = 0
        target = []
        result = []
        print("Lenght curr : ", len(self.current_layer))
        for _ in range(self.max_iter):
            error = 0.0
            for index, item in data.iterrows():
                # Prepare input
                self.enqueue_layer(InputLayer([item['sepal_length'], item['sepal_width'], item['petal_length'], item['petal_width']]))
                # Forward andd result
                self.forward_propagation()
                
                target.append([item['Class_1'], item['Class_2'], item['Class_3']])
                print("Target : ", target)
                result.append(self.current_layer[-1].result)
                if self.current_layer[-1].activation_function_name == "relu" or self.current_layer[-1].activation_function_name == "sigmoid":
                    error += lossFunction(target, result, 3)
                elif self.current_layer[-1].activation_function_name == "softmax":
                    for i in range(len(target)):
                        for j in range(len(target[i])):
                            # print("inside : ", result[i][j])
                            if target[i][j] != result[i][j]:
                                error += lossSoftmax(result[i][j])
                print("error : ", error)
                if error < self.error_threshold:
                    break 

                # cleaning layer
                self.deque_layer()

                # Learn with bach_size
                if (index + 1) % self.batch_size == 0 or index == len(data.index):
                    # backpropagation
                    self.back_propagation(target, result)
                    # clearing list and error foreach batch_size
                    target.clear()
                    result.clear()
                    error = 0
            
            if current_iter < self.max_iter:
                break
    def back_propagation(self, arr_target, arr_out):
        for i in range(len(self.current_layer) - 1, -1, -1):
            if i != len(self.current_layer) - 2 and i > 0: # Not input or output layer
                for j in range(len(self.current_layer[i].weight)):
                    for k in range(len(self.current_layer[i].weight[j])):
                        # print("in k range : ", self.current_layer[i].weight)
                        self.current_layer[i].weight[j][k] = self.current_layer[i].update_weight(arr_target, arr_out, 
                        self.current_layer[i].weight[j], self.current_layer[i].result[j], self.current_layer[i].input_value[j], self.learning_rate)
                        # print(self.current_layer[i].weight[j][k])
                # for j in range(len(self.current_layer[i].bias)):
                #     self.current_layer[i].bias = self.current_layer[i].update_bias(arr_target, arr_out, self.current_layer[i].result[j], self.current_layer[i].input_value[j])
            elif i == len(self.current_layer):
                self.current_layer[i].weight = self.current_layer[i].update_weight_output(arr_target, arr_out, 
                        self.current_layer[i].weight, self.current_layer[i].result[j], self.current_layer[i].input_value[j], self.learning_rate)

    def predict(self, data):
        result = []
        target = []
        precise = 0
        total_data = len(data.index)
        for index, item in data.iterrows():
            # Prepare input
            self.enqueue_layer(InputLayer([item['sepal_length'], item['sepal_width'], item['petal_length'], item['petal_width']]))
            # Forward andd result
            self.forward_propagation()
            target.append([item['Class_1'], item['Class_2'], item['Class_3']])
            result.append(self.current_layer[-1].result)
            max_index_col_result = np.argmax(result[-1], axis=0)
            max_index_col_data = np.argmax(target[-1], axis=0)
            # print("max data ", max_index_col_data)
            # print("max index ", max_index_col_result)
            # print(target)
            # print(result)
            if(max_index_col_data == max_index_col_result): precise = precise + 1
            self.deque_layer()
        accuracy = 0.0
        accuracy = float (precise / total_data)
        print("Accuracy \t: ", accuracy)
             
class InputLayer:
    def __init__(self, arr=[]):
        self.input_value = np.array(arr)
        self.result = self.input_value

    def compute(self):
        pass

from chainRule import *
class Layer(InputLayer):
    def __init__(self, neuron_input, neuron_output, activation_function, activation_function_name, **kwargs):
        super().__init__([])
        self.weight = np.array([[1.5 * (1.0 - random.random()) for x in range(neuron_output)] for j in range(neuron_input)])
        self.bias = np.array([1.5 * (1.0 - random.random()) for x in range(neuron_output)])
        self.result = np.array([])
        self.activation_function = activation_function
        self.activation_function_name = activation_function_name
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
    
    def update_weight(self, arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i, learning_rate):
        return chainRuleHidden(arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i, self.activation_function_name) * learning_rate * -1

    
    def update_weight_output(self, arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i, learning_rate):
        return chainRuleOutput2(arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i, self.activation_function_name) * learning_rate * -1


    def update_bias(self, arr_target, arr_out_o, out_h, vector_i):
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
    layer = []
    learning_rate, error_threshold, max_iter, batch_size = \
        parameter["learning_rate"], parameter["error_threshold"], parameter["max_iter"], parameter["batch_size"]

    # Create base layer
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

        # Case for near Input Layer or the Output Layer
        if index == 0: layer.append(Layer(NEURON_INPUT, item['neuron'], act, item['activation'], threshold=0.1))
        elif index > 0 and index != len(model.index): layer.append(Layer(model.iloc[index - 1, 0], item['neuron'], act, item['activation'], threshold=0.1))
        elif index == model.index: layer.append(OutputLayer(model.iloc[index - 1, 0], item['neuron'], act, item['activation'], threshold=0.1))
    
    print("")

    # Build ANN model from layer and learn process
    neural_network = NeuralNetwork(layer, learning_rate, error_threshold, max_iter, batch_size)
    neural_network.learn(data)
    
    # print("Target Class \t: ", target)
    # print("Predict Class \t: ", result)
    # print("=================================")
    # if (result == target):
    #     print("Result : Good Predict")
    # else:
    #     print("Result : Wrong Predict")
    neural_network.predict(data)
    neural_network.draw()

if __name__ == "__main__":
    main()
    # print(data)
    # print(target)
    # readWeight()
