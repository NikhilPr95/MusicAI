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

	sorted_musicFiles = sorted(musicFiles, key=lambda x: num_bars([x]))

	# remaining_files = []
	# while sorted_musicFiles:
	# 	if sorted_musicFiles:
	# 		remaining_files.append(sorted_musicFiles.pop(0))
	# 	if sorted_musicFiles:
	# 		remaining_files.append(sorted_musicFiles.pop(len(sorted_musicFiles)-1))

	remaining_files = sorted_musicFiles[::-1]

	print('total bars:', total_bars)
	print([num_bars([f]) for f in remaining_files])
	for i in range(n):
		j = 0

		test_set = remaining_files[:j]
		test_bar_val = test_bars
		if n == 5:
			test_bar_val = 0.93*test_bars
			if total_bars <= 500:
				test_bar_val = 0.87*test_bars
		elif n == 10:
			test_bar_val = 0.855*test_bars
			# test_bar_val = 0.9*test_bars
			if total_bars <= 500:
				test_bar_val = 0.8*test_bars
			if total_bars <= 200:
				test_bar_val = 0.65*test_bars

		print('test_bar_val:', test_bar_val)
		while num_bars(test_set) < test_bar_val and j <= len(remaining_files):
			j += 1
			test_set = remaining_files[:j]

		test_files.append(test_set)
		train_files.append(list(set(musicFiles) - set(test_set)))

		remaining_files = remaining_files[j:]

	print([len(t) for t in train_files], [len(t) for t in test_files])
	print([num_bars(t) for t in train_files], [num_bars(t) for t in test_files])
	for train, test in zip(train_files, test_files):
		print(num_bars(test) / (num_bars(train) + num_bars(test)))

	ratios = [(num_bars(test) / (num_bars(train) + num_bars(test))) for train, test in zip(train_files, test_files)]

	assert [len(t) > 0 for t in test_files]

	print('out-', n, sum([num_bars(t) for t in test_files]))

	for train, test in zip(train_files, test_files):
		if n == 5:
			if total_bars < 200:
				assert num_bars(test) / (num_bars(train) + num_bars(test)) >= 0.13
				assert num_bars(test) / (num_bars(train) + num_bars(test)) <= 0.28
			else:
				assert num_bars(test)/(num_bars(train) + num_bars(test)) >= 0.15
				assert num_bars(test) / (num_bars(train) + num_bars(test)) <= 0.22
		elif n == 10:
			if total_bars < 200:
				assert num_bars(test) / (num_bars(train) + num_bars(test)) >= 0.06
				assert num_bars(test) / (num_bars(train) + num_bars(test)) <= 0.16
			else:
				assert num_bars(test) / (num_bars(train) + num_bars(test)) >= 0.06
				assert num_bars(test) / (num_bars(train) + num_bars(test)) <= 0.135

	return train_files, test_files, ratios


def intra_song_splits(original_dir, train_dir, test_dir):
	# Splits songs from the original directory as the first 80 percent in the train dir
	# and the last 20 percent in the test dir

	if not os.path.exists(train_dir):
		os.mkdir(train_dir)

	if not os.path.exists(test_dir):
		os.mkdir(test_dir)

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
