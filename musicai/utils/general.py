import numpy as np


def flatten(list):
	return [l for li in list for l in li]


def make_nparray_from_dict(dic):
	return np.array([[dic[key][genre] for genre in sorted(dic[key])] for key in sorted(dic)]), sorted(dic), \
					sorted(dic[list(dic.keys())[0]])

