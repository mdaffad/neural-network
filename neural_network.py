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
    
    def deque_layer(self):
        self.base_layer.pop(0)
    
    def solve(self):
        self.current_layer = self.base_layer.copy()
        for idx in range(len(self.current_layer)):
            if idx != 0:
                self.current_layer[idx].input_value = self.current_layer[idx-1].result
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
        # case 1 Dimension
        if(len(self.weight) == 1):
            self.result = np.matmul(self.input_value, self.weight.flatten()) + self.bias
        else:
            self.result = np.matmul(np.transpose(self.input_value), self.weight) + self.bias
        
    def compute(self):
        self.sigma()
        self.activate()
        
# driver test
def main():  
    layer = []
    layer.append(InputLayer([1.0, 0.0]))
    import math
    layer.append(Layer([[20, -20], [20,-20]], [-10,30], relu, max_value=0.1)) 
    layer.append(Layer([[20, 20]], [-30], relu,  max_value=0.1)) 
    neural_network = NeuralNetwork()
    neural_network.base_layer = layer
    neural_network.solve()
    for x in neural_network.current_layer:
        print(x.result)
            
    neural_network.deque_layer()
    neural_network.base_layer.insert(0,InputLayer([0.0, 0.1]))
if __name__ == "__main__":
    main()