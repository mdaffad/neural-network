import numpy as np
from activation.activationFunction import linear, sigmoid, relu, softmax

# For iterating per item in array. Except for linear function because single parameter is automate to iterate per item
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
    
    def deque_layer(self, layer):
        self.base_layer.pop(0)
    
    def solve(self):
        self.current_layer = self.base_layer.copy()
        for idx in range(len(self.current_layer)):
            if idx != 0:
                self.current_layer[idx].input_value = self.current_layer[idx-1].result
                # print(self.current_layer[idx].input_value)

            # print(idx)
            self.current_layer[idx].compute()
        
class InputLayer:
    def __init__(self, arr = []):
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
        # print(self.input_value)
        # print((self.weight))
        # print(self.bias)
        if(len(self.weight) == 1):
            self.result = np.matmul(self.input_value, self.weight.flatten())
        else:
            self.result = np.matmul(np.transpose(self.input_value), self.weight) + self.bias
        # print(self.result)
        
        
    def compute(self):
        self.sigma()
        self.activate()
        # print(self.result)
        
# driver test
def main():  
    layer = []
    layer.append(InputLayer([0.0, 0.0]))
    import math
    layer.append(Layer([[20, -20], [20,-20]], [-10,30], sigmoid, threshold=0)) 
    layer.append(Layer([[20, 20]], [-30], sigmoid,  threshold=0)) 
    neural_network = NeuralNetwork()
    neural_network.base_layer = layer
    neural_network.solve()
    # for x in neural_network.current_layer:
    #     print(x.result)

if __name__ == "__main__":
    main()