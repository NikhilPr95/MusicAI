import csv

import os


def sequence_vectors(csvfilepath):
    def getdata(csvfile, data, labels, maxlen):
        rows = csv.reader(open(csvfile, "r"))

        for row in rows:
            left_note_inputs = row[1].split('-')
            bar = [note_val.split('|')[0] for note_val in left_note_inputs if note_val.split('|')[1] != '0']

            if len(bar) > maxlen:
                maxlen = len(bar)

            data.append(bar)
            labels.append(row[2])

        return maxlen

    data = []
    labels = []
    maxlen = 0

    if os.path.isfile(csvfilepath):
        maxlen = getdata(csvfilepath, data, labels, maxlen)
        
    elif os.path.isdir(csvfilepath):
        for csvfile in os.listdir(csvfilepath):
            print(csvfilepath)
            if csvfile.endswith('csv.formatted'):
                maxlen = getdata(csvfilepath+'/'+csvfile, data, labels, maxlen)

    for bar in data:
        if len(bar) < maxlen:
            bar.extend([0]*(maxlen-len(bar)))
    
    return data, labels
