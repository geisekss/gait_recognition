import os
from settings import *
from functions import *

def loadData(path, subset, train_session, test_session):
	"""
	Loads the sessions of a given dataset and divides each session in frames constructing the training and test sets

	Parameters
    ----------
		path: `string` path for the directory where is the database
		subset: `string` name of the specific dataset to load
		train_set: `string` name of which session of the dataset is for training
		test_set: `string` name of which session of the dataset is for test

	Returns
    -------
    	train_set: [M, `frame_size`, `n_accelerations`] `ndarray` the frames for training given that M is the total number of frames obtained from the training sessions
    	train_meta: [M, `dim_meta`] `ndarray` meta information about the training frames as user id of each frame
    	test_set: [L, `frame_size`, `n_accelerations`] `ndarray` the frames for test given that L is the total number of frames obtained from the test sessions
    	test_meta: [L, `dim_meta`] `ndarray` meta information about the test frames as user id of each frame 
    """
	print("-- LOADING DATA from "+path+subset)

	files = [path+subset+'/'+f for f in os.listdir(path+subset) if f[0] != '.']
	train_set = np.empty((0, frame_size, n_accelerations))
	test_set = np.empty((0, frame_size, n_accelerations))
	train_meta = np.empty((0, dim_meta))
	test_meta = np.empty((0, dim_meta))

	for idx, f in enumerate(files):
		f_meta = f.split('___')
		user = int(f_meta[0].split('/')[-1])
		session = idx+1
		data = np.loadtxt(f, delimiter=',')
		data = data[500:-500]
		frames = [data[i:i+frame_size, :] for i in range(0,data.shape[0]-frame_size,step)]
		frames = np.dstack(frames)
		frames = np.rollaxis(frames,-1)
		user_id = np.full((frames.shape[0]), user)
		session_id = np.full((frames.shape[0]), session)
		meta = np.empty((frames.shape[0], dim_meta))
		meta[:, label_user] = user_id
		meta[:, label_session] = session_id
		if(f_meta.__contains__(train_session)):
			train_set = np.vstack((train_set, frames))
			train_meta = np.vstack((train_meta, meta))
		elif(f.split('___').__contains__(test_session)):
			test_set = np.vstack((test_set, frames))
			test_meta = np.vstack((test_meta, meta))
	return [train_set, train_meta, test_set, test_meta]


def sampling(train_set, train_meta, klass, label, n_samples_pos, rate_neg, fold, path_idxs):
	"""
	Down sampling of the training set  

	Parameters
    ----------
    	train_set: [M, N] `ndarray` the M samples of training with N dimension (number of features) 
    	train_meta: [M, `dim_meta`] `ndarray` meta information about the training samples as user id of each frame
		klass: `int` or `float` label of the interested class to be the positive class
		label: `int` column of the train_meta that has the information about the classes
		n_samples_pos: `int` number of samples to have in the positive class after the down sampling
		rate_neg: `float` the proportion relative to number of positive samples to have in the negative class after the down sampling
		fold: `string` folder of the fold createad with the sampling
		path_idxs: `string` path to store the indexes sampled 

	Returns
    -------
    	train_set: [K, N] `ndarray` the sampled training set where ``K = n_samples_pos + (rate_neg * n_samples_pos)`` 
    	train_meta: [K, `dim_meta`] `ndarray` meta information about the training samples as user id of each sample
    
    """
	print('-- SAMPLING TRAINNING')
	directory_idxs = path_idxs+fold+'/'+str(int(klass))+'/'
	if(os.path.isdir(directory_idxs)):
		print('loading indexes...')
		idxs_class_pos = np.loadtxt(directory_idxs+'idxs_pos_train.txt', dtype=int)
		idxs_class_neg = np.loadtxt(directory_idxs+'idxs_neg_train.txt', dtype=int)
	else:
		idxs_class_pos = (train_meta[ : , label] == klass).nonzero()[0]
		idxs_class_neg = (train_meta[ : , label] != klass).nonzero()[0]
		if(n_samples_pos < len(idxs_class_pos)):
			idxs_class_pos = np.random.choice(idxs_class_pos, n_samples_pos)
		idxs_class_neg = np.random.choice(idxs_class_neg, int(n_samples_pos*rate_neg))
		print('saving indexes...')
		os.makedirs(directory_idxs)
		np.savetxt(directory_idxs+'idxs_pos_train.txt', idxs_class_pos, fmt='%d')
		np.savetxt(directory_idxs+'idxs_neg_train.txt', idxs_class_neg, fmt='%d')

	train_set = np.vstack((train_set[idxs_class_pos], train_set[idxs_class_neg]))
	train_meta = np.vstack((train_meta[idxs_class_pos], train_meta[idxs_class_neg]))
	train_meta[:, label] = 1
	train_meta[len(idxs_class_pos):, label] = -1
	return [train_set, train_meta]





