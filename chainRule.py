import numpy as np
import activation.activationFunction
def chainRuleOutputSigmoid (target, out_o) :
    return -( target - out_o ) * out_o * ( 1 - out_o )
def chainRuleOutputRelu (target, out_o):
    return -( target - out_o ) * activation.activationFunction.reluDerivative(out_o)
def chainRuleOutput2 (target, out_h, out_o, method) :
    output = chainRuleOutputSigmoid(target, out_o)
    return output * out_h

def chainRuleHidden (arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i, method) :
    sum_Output = 0
    if method == "sigmoid":
        for j in range(len(arr_target)):
            arr = []
            output = chainRuleOutputSigmoid(arr_target[j], arr_out_o[j])
            arr.append(output)
            # print("target : ",arr_target)
            # print(arr_hiddenLayer_weight[j])
            result = np.prod(arr) * arr_hiddenLayer_weight[j]
            sum_Output += result
        return sum_Output * out_h * ( 1 - out_h ) * vector_i
    elif method == "relu":
        for j in range(len(arr_target)):
            arr = []
            output = 1
            arr.append(output)
            # print("target : ",arr_target)
            # print(arr_hiddenLayer_weight[j])
            result = np.prod(arr) * arr_hiddenLayer_weight[j]
            sum_Output += result
        return sum_Output * out_h * ( 1 - out_h ) * vector_i
    elif method == "relu":
        for j in range(len(arr_target)):
            arr = []
            output = chainRuleOutputRelu(arr_target[j], arr_out_o[j])
            arr.append(output)
            # print("target : ",arr_target)
            # print(arr_hiddenLayer_weight[j])
            result = np.prod(arr) * arr_hiddenLayer_weight[j]
            sum_Output += result
        return sum_Output * out_h * ( 1 - out_h ) * vector_i
    elif method == "softmax":
        for j in range(len(arr_target)):
            arr = []
            for k in range(len(arr_target[j])):
                output = chainSoftMax(arr_target[j][k], arr_out_o[j][k], activation.activationFunction.softmax(arr_out_o[j]), out_h)
                arr.append(output)
            # print("target : ",arr_target)
            # print(arr_hiddenLayer_weight[j])
            result = np.prod(arr) * arr_hiddenLayer_weight[j]
            sum_Output += result
        return sum_Output * vector_i

def chainSoftMax (target, j, probJ, out_h) :
    if (target == j):
        return -( 1 - probJ ) * out_h
    else:
        return probJ * out_h

if __name__ == "__main__":
    arr_target = [0.01, 0.99]
    arr_out_o = [0.7514, 0.7729]
    out_h = 0.5933
    vector_i = 0.05
    arr_hiddenLayer_weight = [0.4, 0.5]

    tes = chainRuleOutput2(arr_target[0], out_h, arr_out_o[0])
    print(tes)

    tes2 = chainRuleHidden(arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i)
    print(tes2)