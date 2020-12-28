from settings import *

def min_max(train_set, test_set):
	"""
	Min-max normalization

	Parameters
    ----------
		train_set: [M, N] `ndarray` the K samples of training with N dimension (number of features) 
		test_set: [L, N] `ndarray` the L samples of test with N dimension (number of features)

	Returns
    -------
    	train_set: [M, N] `ndarray` normalized training set 
		test_set: [L, N] `ndarray` normalized test set
    """

	minimals = np.min(train_set, axis=0)
	maximals = np.max(train_set, axis=0)
	train_set = (train_set - minimals)/(maximals - minimals)
	test_set = (test_set - minimals)/(maximals - minimals)
	return train_set, test_set

def z_norm(train_set, test_set, **kwargs):
   """
	z-norm

	Parameters
    ----------
		train_set: [M, N] `ndarray` the K samples of training with N dimension (number of features) 
		test_set: [L, N] `ndarray` the L samples of test with N dimension (number of features)

	Returns
    -------
    	train_set: [M, N] `ndarray` normalized training set 
	test_set: [L, N] `ndarray` normalized test set
    """
        means = np.mean(train_set, axis=0)
        stds  = np.std(train_set, axis=0)
        train_set = (train_set - means)/(stds + delta)
        test_set = (test_set - means)/(stds + delta)
        return train_set, test_set


def diff(sequence):
   """
	Signal amplitude

	Parameters
    ----------
		sequence: [N,] 1D `ndarray` contaning N samples 
	Returns
    -------
    		amplitude: `float` difference between max and min of the sequence 
    """
	amplitude = np.max(sequence)-np.min(sequence)
	return(amplitude)
