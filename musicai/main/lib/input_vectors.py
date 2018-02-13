import csv

import os


def sequence_vectors(csvfilepath):
    def getdata(csvfile, data, labels, maxlen):
        rows = csv.reader(open(csvfile, "r"))

        for row in rows:
            left_note_inputs = row[1].split('-')
            bar = [note_val.split('|')[0] for note_val in left_note_inputs]

            if len(bar) > maxlen[0]:
                maxlen = [len(bar)]

            data.append(bar)
            labels.append(row[2])

    data = []
    labels = []
    maxlen = [0]

    if os.path.isfile(csvfilepath):
        getdata(csvfilepath, data, labels, maxlen)

    elif os.path.isdir(csvfilepath):
        for csvfile in os.listdir(csvfilepath):
            if csvfile.endswith('csv.formatted'):
                getdata(csvfilepath+'/'+csvfile, data, labels, maxlen)

    for bar in data:
        if len(bar) < maxlen[0]:
            bar.extend([0]*(maxlen[0]-len(bar)))

    return data, labels
