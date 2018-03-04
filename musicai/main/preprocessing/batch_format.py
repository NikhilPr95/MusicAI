import os
import glob
from musicai.main.constants.directories import *

try:
    list(map(lambda song: os.system("python3 " + PREPROCESSING +"splitting.py " + song),
             glob.glob(os.path.join(RAW,"*.csv"))))

    list(map(lambda song: os.system("python3 " + PREPROCESSING +"chord_creation.py " + song),
             glob.glob(os.path.join(PROCESSED,"*.csv.formatted"))))
except:
    list(map(lambda song: os.system("python " + PREPROCESSING +"splitting.py " + song),
             glob.glob(os.path.join(RAW,"*.csv"))))

    list(map(lambda song: os.system("python " + PREPROCESSING +"chord_creation.py " + song),
             glob.glob(os.path.join(PROCESSED,"*.csv.formatted"))))
