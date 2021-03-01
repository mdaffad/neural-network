import csv

with open("xor-data.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row["x0"], row["x1"], row["x2"], row["f"])

with open("xor-weight.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(
            row["layerFrom"],
            row["from"],
            row["layerTo"],
            row["to"],
            row["weight"],
        )
