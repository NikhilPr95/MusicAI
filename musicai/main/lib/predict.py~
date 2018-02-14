# facade for both knn and omm
# is called by input.py
# completes prediction depending upon algorithm and
# writes to shared memory

from lib.knn import knn_predict
from lib.omm import omm_predict

def predict(prev_bar):
	if len(prev_bar) < 10:
		pass 	# pad with zeros
	if len(prev_bar) > 10:
		pass	# crop perhaps	
	
	# push to shared memory instead of returning here	
	return omm_predict(knn_predict(prev_bar)[0])
	
