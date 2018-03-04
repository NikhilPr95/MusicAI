# facade for both knn and omm
# is called by input.py
# completes prediction depending upon algorithm and
# writes to shared memory

from musicai.main.lib.knn import knn_predict
from musicai.main.lib.markov import omm_predict

def predict(prev_bar):
	MAX_NOTES = 15
	
	if len(prev_bar) < MAX_NOTES:
		prev_bar += [0] * (MAX_NOTES - len(prev_bar)) 	# pad with zeros
	if len(prev_bar) > MAX_NOTES:
		prev_bar = prev_bar[:MAX_NOTES]	# crop perhaps	
	
	# push to shared memory instead of returning here
	return omm_predict(knn_predict(prev_bar)[0])
	
