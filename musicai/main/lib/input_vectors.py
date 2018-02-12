import csv


def sequence_vectors(csvfilepath):
    rows = csv.reader(open(csvfilepath, "r"))

    data = []
    labels = []

    maxlen = 0

    for row in rows:
        left_note_inputs = row[1].split('-')
        bar = [note_val.split('|')[0] for note_val in left_note_inputs]

        if len(bar) > maxlen:
            maxlen = len(bar)

        data.append(bar)
        labels.append(row[2])

    for bar in data:
        if len(bar) < maxlen:
            bar.extend([0]*(maxlen-len(bar)))

    return data, labels
