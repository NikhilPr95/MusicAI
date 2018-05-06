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

