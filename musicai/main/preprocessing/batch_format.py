import os
import glob
from musicai.main.constants.directories import *

'''
RAW -> PROCESSED -> PROCESSED_CHORDS

RAW -> RAW_TIME -> PROCESSED_TIME -> PROCESSED_CHORDS_TIME
'''

try:
    list(map(lambda song: os.system("python3 " + PREPROCESSING +"splitting.py " + song),
             glob.glob(os.path.join(RAW,"*.csv"))))

    list(map(lambda song: os.system("python3 " + PREPROCESSING +"chord_creation.py " + song),
             glob.glob(os.path.join(PROCESSED,"*.csv.formatted"))))

    list(map(lambda song: os.system("python3 " + PREPROCESSING + "add_duration.py " + song),
             glob.glob(os.path.join(RAW,"*.csv"))))

    list(map(lambda song: os.system("python3 " + PREPROCESSING +"splitting.py " + song + " --time"),
             glob.glob(os.path.join(RAW_TIME,"*.csv"))))

    list(map(lambda song: os.system("python3 " + PREPROCESSING +"chord_creation.py " + song  + " --time"),
             glob.glob(os.path.join(PROCESSED_TIME,"*.csv.formatted"))))

except:
    list(map(lambda song: os.system("python " + PREPROCESSING +"splitting.py " + song),
             glob.glob(os.path.join(RAW,"*.csv"))))

    list(map(lambda song: os.system("python " + PREPROCESSING +"chord_creation.py " + song),
             glob.glob(os.path.join(PROCESSED,"*.csv.formatted"))))

    list(map(lambda song: os.system("python " + PREPROCESSING + "add_duration.py " + song),
             glob.glob(os.path.join(RAW,"*.csv"))))

    list(map(lambda song: os.system("python " + PREPROCESSING +"splitting.py " + song + "--time"),
             glob.glob(os.path.join(RAW_TIME,"*.csv.formatted"))))

    list(map(lambda song: os.system("python " + PREPROCESSING +"chord_creation.py " + song  + "--time"),
             glob.glob(os.path.join(PROCESSED_TIME,"*.csv.formatted"))))

