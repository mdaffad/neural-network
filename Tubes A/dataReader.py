import csv


def readData(filename="xor-data.csv"):
    data = []
    target_class = []
    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append([int(row["x1"]), int(row["x2"])])
            target_class.append(int(row["f"]))
    return data, target_class


def readWeight(filename):
    f = open(filename, "r")
    f1 = f.readlines()
    activation = []
    bias = []
    weight = []
    allowed = ['relu', 'sigmoid', 'linear', 'softmax']

    n_attr = int(f1[0])
    line = 1
    while (line < len(f1)):
        splitted = f1[line].strip('\n').split(' ')
        if (('relu' in splitted) or ('linear' in splitted) or ('sigmoid' in splitted) or ('softmax' in splitted)):
            activation.append(splitted[1])

            line += 1
            splitted = f1[line].strip('\n').split(' ')

            bias.append(list(map(int, splitted)))
        else:
            temp = []
            count = 0
            while (count < n_attr):
                splitted = f1[line].strip('\n').split(' ')
                temp.append(list(map(int, splitted)))
                count += 1
                line += 1
            if (count == n_attr):
                weight.append(temp)
                line -= 1
        line += 1
    return activation, bias, weight
