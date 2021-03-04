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

def linear(x, kwargs=None):
    return x

def sigmoid(x, kwargs=None):
    value = float(1 / (1 + math.exp(x * -1)))
    threshold = kwargs.get("threshold", None)
    if threshold == None:
        return value
    else:
        if value < threshold:
            return 0
        else:
            return 1

def relu(x, kwargs):
    alpha = kwargs.get("alpha", 0.0)
    max_value = kwargs.get("max_value", None)
    threshold = kwargs.get("threshold", 0.0)
    if x < threshold:
        return max(x, x * alpha)
    else:
        if max_value == None:
            return x
        else:
            return min(x, max_value)

def softmax(arr):
    arr_exp = np.exp(arr)
    return arr_exp / arr_exp.sum()
