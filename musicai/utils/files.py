import csv
import glob

import os
import random
from copy import copy

from musicai.main.constants.values import BAR_THRESHOLD


def num_bars(files):
	return sum([len(open(f).readlines()) for f in files])


def randomSplit(directory):
	musicFiles_ = glob.glob(os.path.join(directory, "*"))
	random.shuffle(musicFiles_)
	musicFiles = [f for f in musicFiles_ if len(open(f).readlines()) > BAR_THRESHOLD]
	length = len(musicFiles)

	trainData = musicFiles[:int(0.8 * length)] + list(set(musicFiles_) - set(musicFiles))
	train_bars = sum([len(open(f).readlines()) for f in trainData])
	testData = musicFiles[int(0.8 * length):]
	test_bars = sum([len(open(f).readlines()) for f in testData])

	print("------")
	print(len(trainData), '(', train_bars, ')', len(testData), '(', test_bars, ')')
	print("-----")

	return trainData, testData


def get_num_splits(train_files, test_files):
	return [(num_bars(x), num_bars(y), num_bars(x)/(num_bars(x) + num_bars(y))) for x,y in zip(test_files,train_files)]


def kfold_split(directory, n):
	musicFiles = sorted(glob.glob(os.path.join(directory, "*")))
	total_bars = num_bars(musicFiles)

	test_bars = total_bars/n

	train_files = []
	test_files = []

	remaining_files = musicFiles
	for i in range(n):
		j = 0

		test_set = remaining_files[:j]
		while num_bars(test_set) < 0.95*test_bars and j <= len(remaining_files):
			j += 1
			test_set = remaining_files[:j]

		test_files.append(test_set)
		train_files.append(list(set(musicFiles) - set(test_set)))

		remaining_files = remaining_files[j:]

	return train_files, test_files


def intra_song_splits(original_dir, train_dir, test_dir):
	# Splits songs from the original directory as the first 80 percent in the train dir
	# and the last 20 percent in the test dir

	for csvfile in os.listdir(original_dir):
		print(csvfile)
		rows = csv.reader(open(os.path.join(original_dir, csvfile), "r"))
		rows = copy(list(rows))

		train_dir_writer = csv.writer(open(os.path.join(train_dir, csvfile), "w"), lineterminator='\n')
		for row in rows[:int(0.8*len(rows))]:
			train_dir_writer.writerow(row)

		test_dir_writer = csv.writer(open(os.path.join(test_dir, csvfile), "w"), lineterminator='\n')

		for row in rows[int(0.8*len(rows)):]:
			test_dir_writer.writerow(row)
