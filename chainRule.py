import numpy as np

def chainRuleOutput1 (target, out_o) :
    return -( target - out_o ) * out_o * ( 1 - out_o )

def chainRuleOutput2 (target, out_h, out_o) :
    output = chainRuleOutput1(target, out_o)
    return output * out_h

def chainRuleHidden (arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i) :
    sum_Output = 0
    for i in arr_hiddenLayer_weight:
        arr = np.array([])
        for j in arr_target:
            output = chainRuleOutput1(arr_target[j], arr_out_o[j])
            arr = np.append(output)
        result = np.prod(arr) * arr_hiddenLayer_weight[i]
        sum_Output += result
    return sum_Output * out_h * ( 1 - out_h ) * vector_i

# def chainSoftMax () :
#     return 