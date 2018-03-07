from musicai.main.constants import directories
import glob, random
from musicai.main.models.ko import KO
from musicai.main.lib.input_vectors import sequence_vectors, parse_data
from musicai.tests.metrics import percentage
from musicai.utils.general import *
import os
def splitData():

    musicFiles = glob.glob(os.path.join(directories.PROCESSED_CHORDS,"*"))
    length = len(musicFiles)

    random.shuffle(musicFiles)
    trainData = musicFiles[:int(0.6*length)]
    valData = musicFiles[int(0.6 * length):int(0.8 * length)]
    testData = musicFiles[int(0.8 * length):]

    return trainData, valData, testData

def fitModel(option, train):
    obj = None
    '''
    data = sequence_vectors(train, padding=15)
    X = data[0]
    y = data[1]
    X = flatten(X)
    y = flatten(y)
    lengths = [len(x) for x in y]
    '''
    if option == 1:
        obj = KO()
        print(train)
        obj.fit(train)

    elif option == 2:
        pass

    elif option == 3:
        pass


    return obj

def testModel():
    print("Your options for the model : ")
    print(" 1) KNN and OMM")
    print(" 2) HMM ")
    print(" 3) RNN (Coming soon) :) ")
    option = int(input("Enter your choice : "))

    dataset = splitData()
    test = dataset[2]
    songs_bar_sequences,songs_chord_sequences = [] ,[]
    for file_name in test:
        data = sequence_vectors(file_name)
        songs_bar_sequences.append(data[0])
        songs_chord_sequences.append(data[1])


    if option == 1:
        obj = fitModel(option, dataset[0])
        predicted_knn, predicted_omm = [], []
        for bar_sequences in songs_bar_sequences:
            predicted_knn.append([])
            predicted_omm.append([])
            for bar_sequence in bar_sequences[:-1]: #this is required as the previous bar is sent to omm, hence we need to remove the last bar.
                result = obj.predict(bar_sequence)
                predicted_knn[-1].append(result[0])
                predicted_omm[-1].append(result[1])

        #the predicted and song_chord_sequences is for each song, so we have to iterate through the song and combine all values to one list
        perc_knn = percentage(flatten(songs_chord_sequences),flatten(predicted_knn))
        return perc_knn, percentage([chord_sequence for chord_sequences in songs_chord_sequences for chord_sequence in chord_sequences[1:]],flatten(predicted_omm))
    elif option == 2:
        pass

    elif option == 3:
        print("Haha! You thought null pointer, didn't you? \n Coming soon!\n\n\n\n The RNN, not the null pointer")


if __name__ == "__main__":
    print(testModel())