import os
import glob
list(map(lambda song : os.system("python3 chord_creation.py " + song),glob.glob("../../data/processed/*.csv.formatted")))

