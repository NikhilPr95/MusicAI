import os
import glob
list(map(lambda song : os.system("python3 splitting.py " + song),glob.glob("../../data/raw/*.csv")))

