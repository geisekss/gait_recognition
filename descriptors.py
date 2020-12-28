from settings import *
import pywt

def resultant_vector(train_set, test_set):
	"""
	Calculates the magnitude/resultant vector by `r = sqrt(x^2 + y^2 + z^2)` for each frame

	Parameters
    ----------
		train_set: [M, `frame_size`, `n_accelerations`] `ndarray` frames for training given that M is the total number of frames obtained from the training sessions
		test_set: [L, `frame_size`, `n_accelerations`] `ndarray` frames for test given that L is the total number of frames obtained from the test sessions

	Returns
    -------
    	train_set: [M, `frame_size`] `ndarray` magnitude or resultant vector of each frame of training
		test_set: [L, `frame_size`] `ndarray` magnitude or resultant vector of each frame of test 
    """
	features_train = np.sqrt(np.sum(train_set**2, axis=2))
	features_test = np.sqrt(np.sum(test_set**2, axis=2))
	return features_train, features_test

def time(train_set, test_set):
	"""
	Obtains the feature vector in the time domain for each frame

	Parameters
    ----------
		train_set: [M, `frame_size`, `n_accelerations`] `ndarray` frames for training given that M is the total number of frames obtained from the training sessions
		test_set: [L, `frame_size`, `n_accelerations`] `ndarray` frames for test given that L is the total number of frames obtained from the test sessions

	Returns
    -------
    	train_set: [M, `n_accelerations * frame_size`] `ndarray` accelerations concatenated to represent a feature vector for each frame of the training
		test_set: [L, `n_accelerations * frame_size`] `ndarray` accelerations concatenated to represent a feature vector for each frame of the test 
    """
	features_train = train_set.reshape(train_set.shape[0], -1)
	features_test = test_set.reshape(test_set.shape[0], -1)
	return features_train, features_test

def freq(train_set, test_set):
	"""
	Generates the feature vector in the frequency domain for each frame

	Parameters
    ----------
		train_set: [M, `frame_size`, `n_accelerations`] `ndarray` frames for training given that M is the total number of frames obtained from the training sessions
		test_set: [L, `frame_size`, `n_accelerations`] `ndarray` frames for test given that L is the total number of frames obtained from the test sessions

	Returns
    -------
    	train_set: [M, `n_accelerations * frame_size`] `ndarray` frequency magnitudes of each acceleration concatenated to represent a feature vector for each frame of the training
		test_set: [L, `n_accelerations * frame_size`] `ndarray` frequency magnitudes of each acceleration concatenated to represent a feature vector for each frame of the test 
    """
	n_freq = int(train_set.shape[1]/2)
	fft_coeffs_train = np.array([np.array([np.abs(np.fft.fft(train_set[i, :, k]))[:n_freq] for k in range(train_set.shape[2])]).T for i in range(train_set.shape[0])])
	fft_coeffs_test = np.array([np.array([np.abs(np.fft.fft(test_set[i, :, k]))[:n_freq] for k in range(test_set.shape[2])]).T for i in range(test_set.shape[0])])
	features_train = fft_coeffs_train.reshape(fft_coeffs_train.shape[0], -1)
	features_test = fft_coeffs_test.reshape(fft_coeffs_test.shape[0], -1)
	return features_train, features_test

def wavelet(train_set, test_set, level=6):
   """
	Generates the feature vector in the time-frequency (wavelet) domain for each frame

	Parameters
    ----------
		train_set: [M, `frame_size`, `n_accelerations`] `ndarray` frames for training given that M is the total number of frames obtained from the training sessions
		test_set: [L, `frame_size`, `n_accelerations`] `ndarray` frames for test given that L is the total number of frames obtained from the test sessions

	Returns
    -------
    	train_set: [M, `n_accelerations * frame_size`] `ndarray` wavelet coeffcients of each acceleration concatenated to represent a feature vector for each frame of the training
		test_set: [L, `n_accelerations * frame_size`] `ndarray`  wavelet coeffcients of each acceleration concatenated to represent a feature vector for each frame of the test 
    """
	wavelet_coeffs_train = np.array([np.array([np.hstack((pywt.wavedec(train_set[i, :, k], 'haar', level=level))) for k in range(train_set.shape[2])]).T for i in range(train_set.shape[0])])
	wavelet_coeffs_test = np.array([np.array([np.hstack((pywt.wavedec(test_set[i, :, k], 'haar', level=level))) for k in range(test_set.shape[2])]).T for i in range(test_set.shape[0])])
	features_train = wavelet_coeffs_train.reshape(wavelet_coeffs_train.shape[0], -1)
	features_test = wavelet_coeffs_test.reshape(wavelet_coeffs_test.shape[0], -1)
	return features_train, features_test

def histogram(train_set, test_set):
    """
	Generates the feature vector of a 10-bin histogram

	Parameters
    ----------
		train_set: [M, `frame_size`, `n_accelerations`] `ndarray` frames for training given that M is the total number of frames obtained from the training sessions
		test_set: [L, `frame_size`, `n_accelerations`] `ndarray` frames for test given that L is the total number of frames obtained from the test sessions

	Returns
    -------
    	train_set: [M, `n_accelerations * 10`] `ndarray` 10-bin histogram of each acceleration concatenated to represent a feature vector for each frame of the training
	test_set: [L, `n_accelerations * 10`] `ndarray`  10-bin histogram of of each acceleration concatenated to represent a feature vector for each frame of the test 
    """
	features_train = np.array([np.concatenate(([np.histogram(train_set[i, :, k], bins=10, density=True, range=(-10,10))[0] for k in range(train_set.shape[-1])])) for i in range(train_set.shape[0])])
	features_test = np.array([np.concatenate(([np.histogram(test_set[i, :, k], bins=10, density=True, range=(-10,10))[0] for k in range(test_set.shape[-1])])) for i in range(test_set.shape[0])])
	return features_train, features_test

def statistics(train_set, test_set):
"""
	Generates the feature vector of a 10-bin histogram

	Parameters
    ----------
		train_set: [M, `frame_size`, `n_accelerations`] `ndarray` frames for training given that M is the total number of frames obtained from the training sessions
		test_set: [L, `frame_size`, `n_accelerations`] `ndarray` frames for test given that L is the total number of frames obtained from the test sessions

	Returns
    -------
    	train_set: [M, `n_accelerations * 6`] `ndarray` statistics of each acceleration concatenated to represent a feature vector for each frame of the training
	test_set: [L, `n_accelerations * 6`] `ndarray`  statistics of each acceleration concatenated to represent a feature vector for each frame of the test 
    """
	features_train = np.array([np.concatenate(([[np.min(train_set[i, :, k]), np.max(train_set[i, :, k]), np.mean(train_set[i, :, k]), np.std(train_set[i, :, k]), np.sqrt(np.mean(train_set[i, :, k]**2)), diff(train_set[i, :, k])] for k in range(train_set.shape[-1])])).T for i in range(train_set.shape[0])])
	features_test = np.array([np.concatenate(([[np.min(test_set[i, :, k]), np.max(test_set[i, :, k]), np.mean(test_set[i, :, k]), np.std(test_set[i, :, k]), np.sqrt(np.mean(test_set[i, :, k]**2)), diff(test_set[i, :, k])] for k in range(test_set.shape[-1])])).T for i in range(test_set.shape[0])])
	return features_train, features_test
