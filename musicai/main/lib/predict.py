# facade for both knn and omm
# is called by input.py
# completes prediction depending upon algorithm and
# writes to shared memory

from musicai.main.lib.knn import knn_predict
from musicai.main.lib.markov import omm_predict
from musicai.main.lib.lstm import lstm_predict
from musicai.main.constants.values import *
def predict(prev_bar,model="KO"):
	
	if(model == "KO"):
		if len(prev_bar) < MAX_NOTES:
			prev_bar += [0] * (MAX_NOTES - len(prev_bar)) 	# pad with zeros
		if len(prev_bar) > MAX_NOTES:
			prev_bar = prev_bar[:MAX_NOTES]	# crop perhaps
		
		# push to shared memory instead of returning here
		return omm_predict(knn_predict(prev_bar)[0])

	elif(model == "lstm"):
		if len(prev_bar) < MAX_NOTES:
			prev_bar += [0] * (MAX_NOTES - len(prev_bar)) 	# pad with zeros
		if len(prev_bar) > MAX_NOTES:
			prev_bar = prev_bar[:MAX_NOTES]	# crop perhaps
			
		return lstm_predict(prev_bar)
