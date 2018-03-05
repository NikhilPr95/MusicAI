def percentage(actual, predicted):
	'''
	returns fraction of chords that match in their corresponding positions
	higher this value, better the model
	'''
	return sum(list(map(lambda x,y : int(x == y),actual, predicted)))/len(actual)

def precision(actual, predicted):
	'''
	returns the % of chords in predicted that are present in actual, but not necessarily in the correct position
	this metric checks if the model is at least getting the chords right, if not the position (perhaps due to usage of previous bar notes)
	a high value does not guarantee a good model as the prediction could just be the same chord which happens to be in the actual
	'''
	new_chords =  set(predicted) - set(actual)
	new_chord_count = 0
	if new_chords:
		for chord in predicted:
			if chord in new_chords:
				new_chord_count += 1
	return 1 - (new_chord_count/len(predicted))

def recall(actual, predicted):
	'''
	returns the % of chords in actual that are present in predicted, but not necessarily in the correct position
	'''
	return precision(predicted, actual)

def longest_good_run(actual, predicted, fn = str.__eq__):
	'''
	returns fraction of longest sequence for which chord prediction was correct
	'''
	count = 0
	maxlen = 0
	for pair in zip(actual, predicted):
		if fn(pair[0],pair[1]):
			count += 1
			if maxlen < count:
				maxlen = count
		else:
			count = 0
	return maxlen/actual
def longest_bad_run(actual, predicted):
	'''
	returns fraction of longest sequence for which chord prediction was incorrect
	'''
	longest_good_run(actual, predicted, fn = str.__ne__)
