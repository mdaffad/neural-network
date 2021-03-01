import math
import numpy as np


class Activation:
    def __init__(self, weight=[], vector=[]):

        if len(weight) != len(vector):
            if weight != []:
                raise Exception(
                    "the length between weight vector must be equal or the weight must be empty list"
                )

        self.weight = weight
        self.vector = vector

        if self.weight != []:
            self.sigma = sum(
                [weight[i] * vector[i] for i in range(len(weight))]
            )
        else:
            self.sigma = sum(vector)

    def linear(self):

        return self.sigma

    def sigmoid(self, threshold=None):

        value = float(1 / (1 + math.exp(self.sigma * -1)))
        if threshold == None:
            return value
        else:
            if value < threshold:
                return 0
            else:
                return 1

    def relu(self, alpha=0.0, max_value=None, threshold=0):

        if self.sigma < threshold:
            return max(self.sigma, self.sigma * alpha)
        else:
            if max_value == None:
                return self.sigma
            else:
                return min(self.sigma, max_value)


def softmax(arr):

    arr_exp = np.exp(arr)
    return arr_exp / arr_exp.sum()
