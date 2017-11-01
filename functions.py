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
