from collections import Counter
from settings import *


def max_prob(probabilities, y_test, labels_sessions_test):
	"""
	Later fusion using the method of max probability among a sequencial subset of probabilities with lenght `n_fusion`

	Parameters
    ----------
		probabilities: [L, 2] `ndarray` the classification probabilities/confidences of the test samples in the sklearn format 
		y_test: [L, ] `ndarray` ground truth for the test samples
		labels_sessions_test: [L, ] `ndarray` the sessions id for the test samples

	Returns
    -------
    	y_true: [J, ] `ndarray` ground truth for the test samples after the fusion given that ``J = N - n_fusion``
    	y_pred: [J, ] `ndarray` labels prediction for the test samples based on the probabilities 
    	y_prob: [J, ] `ndarray` the probability relative to each prediction which is the maximum value of the evaluated subset 
    """
	_, idx = np.unique(labels_sessions_test, return_index=True)
	sessions = labels_sessions_test[np.sort(idx)]
	y_prob = np.empty((0,))
	y_pred = np.empty((0,))
	y_true = np.empty((0,))
	for s in sessions:
		idxs_s = np.where(labels_sessions_test==s)[0]
		maximals = np.array([np.max(probabilities[idxs_s[i:i+n_fusion]], axis=0) for i in range(len(idxs_s)-n_fusion)])
		idxs_max = np.argmax(maximals, axis=1)
		predictions = np.hstack((np.full((len(idxs_max), 1), -1), np.ones((len(idxs_max), 1)) ))
		y_prob = np.hstack((y_prob, maximals[np.arange(len(maximals)),idxs_max]))
		y_pred = np.hstack((y_pred, predictions[np.arange(len(maximals)),idxs_max]))
		y_true = np.hstack((y_true, y_test[idxs_s[:-n_fusion]]))	        
	return y_true, y_pred, y_prob


def mean(probabilities, y_test, labels_sessions_test):
	"""
	Later fusion using the method of mean among a sequencial subset of probabilities with lenght `n_fusion`
	We consider the probabilities of the negative class as negative values to calculate the average

	Parameters
    ----------
		probabilities: [L, 2] `ndarray` the classification probabilities/confidences of the test samples in the sklearn format 
		y_test: [L, ] `ndarray` ground truth for the test samples
		labels_sessions_test: [L, ] `ndarray` the sessions id for the test samples

	Returns
    -------
    	y_true: [J, ] `ndarray` ground truth for the test samples after the fusion given that ``J = N - n_fusion``
    	y_pred: [J, ] `ndarray` labels prediction for the test samples based on the probabilities 
    	y_prob: [J, ] `ndarray` the probability relative to each prediction which is the mean of the evaluated subset 
    """
	_, idx = np.unique(labels_sessions_test, return_index=True)
	sessions = labels_sessions_test[np.sort(idx)]
	y_prob = np.empty((0,))
	y_pred = np.empty((0,))
	y_true = np.empty((0,))
	probabilities[:, 0] *= -1
	for s in sessions:
		idxs_s = np.where(labels_sessions_test==s)[0]
		probabilities_windows = [probabilities[idxs_s[i:i+n_fusion]] for i in range(len(idxs_s)-n_fusion)]
		averages = np.array([np.mean(pw[np.arange(pw.shape[0]), np.argmax(np.abs(pw), axis=1)]) for pw in probabilities_windows])
		predictions = np.sign(averages)
		y_prob = np.hstack((y_prob, averages))
		y_pred = np.hstack((y_pred, predictions))
		y_true = np.hstack((y_true, y_test[idxs_s[:-n_fusion]]))
	return y_true, y_pred, y_prob


def voting(probabilities, y_set, labels_sessions_test):
	"""
	Later fusion using the method of majority voting among a sequencial subset of predictions with lenght `n_fusion`
	
	Parameters
    ----------
		probabilities: [L, 2] `ndarray` the classification probabilities/confidences of the test samples in the sklearn format 
		y_test: [L, ] `ndarray` ground truth for the test samples
		labels_sessions_test: [L, ] `ndarray` the sessions id for the test samples

	Returns
    -------
    	y_true: [J, ] `ndarray` ground truth for the test samples after the fusion given that ``J = N - n_fusion``
    	y_pred: [J, ] `ndarray` labels prediction for the test samples based on the probabilities 
    """
	predictions = np.argmax(probabilities, axis=1)
	predictions[np.where(predictions==0)[0]] = -1
	y_pred = np.empty((0,))
	y_true = np.empty((0,))
	for s in np.unique(labels_sessions_test):
		idxs_s = np.where(labels_sessions_test==s)[0]
		predictions_session = np.array([Counter(predictions[idxs_s[i:i+n_fusion]]).most_common()[0][0] for i in range(len(idxs_s)-n_fusion)])
		y_pred = np.hstack((y_pred, predictions_session))
		y_true = np.hstack((y_true, y_test[idxs_s[:-n_fusion]]))
	return y_true, y_pred

   



