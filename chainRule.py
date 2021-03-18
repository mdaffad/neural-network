import numpy as np

def chainRuleOutput1 (target, out_o) :
    return -( target - out_o ) * out_o * ( 1 - out_o )

def chainRuleOutput2 (target, out_h, out_o) :
    output = chainRuleOutput1(target, out_o)
    return output * out_h

def chainRuleHidden (arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i) :
    sum_Output = 0
    for j in range(len(arr_target)):
        arr = []
        output = chainRuleOutput1(arr_target[j], arr_out_o[j])
        arr.append(output)
        result = np.prod(arr) * arr_hiddenLayer_weight[j]
        sum_Output += result
    return sum_Output * out_h * ( 1 - out_h ) * vector_i

# def chainSoftMax () :
#     return 

arr_target = [0.01, 0.99]
arr_out_o = [0.7514, 0.7729]
out_h = 0.5933
vector_i = 0.05
arr_hiddenLayer_weight = [0.4, 0.5]

tes = chainRuleOutput2(arr_target[0], out_h, arr_out_o[0])
print(tes)

tes2 = chainRuleHidden(arr_target, arr_out_o, arr_hiddenLayer_weight, out_h, vector_i)
print(tes2)