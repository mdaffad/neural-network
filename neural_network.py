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
        for idx, layer in enumerate(self.current_layer):
            if idx != 0:
                layer.input_value = self.current_layer[idx-1].result
            layer.compute()
        
class InputLayer:
    def __init__(self, arr = []):
        self.input_value = np.array(arr)
        self.result = self.input_value
    
    def compute(self):
        pass

class Layer(InputLayer):
    def __init__(self, arr_weight, arr_vector, activation_function, arr_input = []):
        super().__init__(arr_input)
        self.weight = np.array(arr_weight)
        self.vector = np.array(arr_vector)
        self.result = np.array([])
        self.activation_function = activation_function
    
    def activate(self):
        self.result = self.activation_function(self.result)
    
    def sigma(self):
        self.result = np.matmul(np.transpose(self.input_value), self.weight) + self.vector
    
    def compute(self):
        self.sigma()
        self.activate()

# driver test
def main():  
    layer = []
    layer.append(InputLayer([0.05, 0.1]))
    import math
    layer.append(Layer([[0.15, 0.2], [0.25, 0.3]], [0.35,0.35], linear)) # result from slide [0.5933, 0.5969] 
    layer.append(Layer([[0.4, 0.45], [0.5, 0.55]], [0.6,0.6], linear)) # result from slide [0.7514, 0.7729]
    neural_network = NeuralNetwork()
    neural_network.base_layer = layer
    neural_network.solve()
    for x in neural_network.current_layer:
        print(x.result)

if __name__ == "__main__":
    main()