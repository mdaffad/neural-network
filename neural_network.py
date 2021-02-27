import numpy as np
class NeuralNetwork:
    def __init__(self):
        self.layer = []
    
    def get_total_layer(self):
        return len(self.layer)

class Layer:
    def __init__(self, activation_function):
        self.weight = np.array([])
        self.bias = np.array([])
        self.result = np.array([])
        self.activation_function = activation_function
    def activate(self):
        self.result = self.activation_function(self.result)
