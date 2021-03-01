import numpy as np

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
    def __init__(self, arr_weight, arr_bias, activation_function, arr_input = []):
        super().__init__(arr_input)
        self.weight = np.array(arr_weight)
        self.bias = np.array(arr_bias)
        self.result = np.array([])
        self.activation_function = activation_function
    
    def activate(self):
        self.result = self.activation_function(self.result)
    
    def compute(self):
        self.result = np.matmul(np.transpose(self.input_value), self.weight) + self.bias
    
    def output(self):
        self.compute()
        self.result()

# driver test
def main():
    input_layer = InputLayer([0,1,2])
    print(type(input_layer.input_value))    
    matrix = np.array([[0,1,2], [3,4,5]])
    print(matrix[0])
    print(np.matmul(matrix, input_layer.input_value))    

if __name__ == "__main__":
    main()