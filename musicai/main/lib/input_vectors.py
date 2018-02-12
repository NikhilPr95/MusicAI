import csv


def sequence_vectors(csvfilepath):
    rows = csv.reader(open(csvfilepath, "r"))
    for row in rows:
        print(row)