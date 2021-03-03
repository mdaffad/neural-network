import csv
import pandas as pd


def readData(filename="xor-data.csv"):
    data = []
    target_class = []
    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append([int(row["x1"]), int(row["x2"])])
            target_class.append(int(row["f"]))
    return data, target_class


def readWeight(filename="xor-weight.csv"):
    # with open("xor-weight.csv", newline="") as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         print(
    #             row["layerFrom"],
    #             row["from"],
    #             row["layerTo"],
    #             row["to"],
    #             row["weight"],
    #         )
    f = open("test.txt", "r")
    f1 = f.readlines()
    activation = []
    bias = []
    weight = []
    for x in f1:
        splitted = x.strip('\n').split(' ')
        print(splitted)
