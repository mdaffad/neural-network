import math
import numpy as np


# class Activation:
#     def __init__(self, weight=[], vector=[]):

#         if len(weight) != len(vector):
#             if weight != []:
#                 raise Exception(
#                     "the length between weight vector must be equal or the weight must be empty list"
#                 )

#         self.weight = weight
#         self.vector = vector

#         if self.weight != []:
#             self.sigma = sum(
#                 [weight[i] * vector[i] for i in range(len(weight))]
#             )
#         else:
#             self.sigma = sum(vector)

def linear(x):
    return x

def sigmoid(x, threshold=None):
    value = float(1 / (1 + math.exp(x * -1)))
    if threshold == None:
        print(value)
        return value
    else:
        if value < threshold:
            return 0
        else:
            return 1

def relu(x, alpha=0.0, max_value=None, threshold=0):
    print(x)
    if x < threshold:
        return max(x, x * alpha)
    else:
        if max_value == None:
            return x
        else:
            return min(x, max_value)

def softmax(arr):
    arr_exp = np.exp(arr)
    print("enter softmax")
    print(arr_exp)
    return arr_exp / arr_exp.sum()
