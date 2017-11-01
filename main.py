import sys
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import temporal_coherence
from temporal_coherence import *
from descriptors import *
from processing_data import *
from settings import *
from functions import *


def trainModel(x_train, y_train, classifier, score):
	"""
	Training of a model given the features and labels of training using a specified classifier

	Parameters
    ----------
		x_train: [M, N] `ndarray` the set with N dimensionality used for training of the classifier 
		y_train: [M, ] `ndarray` the labels corresponding to each example of the training set
		classifier: `string` name of the classifier to be adopted (options implemented: `svm` and `rd_forest`)
		score: `string` which score should be otimized in the grid search, e.g., `accuracy`, `roc_auc`

	Returns
    -------
    	clf: `GridSearchCV object` result of the grid seach with the best parameters trained classifier
    """

	print("-- TRAINNING: grid search with 5 fold cross-validation")

	if(classifier == 'svm'):
		train_set, test_set = min_max(train_set, test_set)
		tuned_parameters = [{
			'kernel': ['rbf', 'linear'], 
			'gamma': [2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 1, 2, 2**2, 2**3, 2**4, 2**5, 2**6],
            'C': [2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 1, 2, 2**2, 2**3, 2**4, 2**5, 2**6]
		}]		
		clf = model_selection.GridSearchCV(svm.SVC(probability=True), tuned_parameters, cv=5, scoring=score, verbose=0, n_jobs=5)
	elif(classifier == "rd_forest"):
		tuned_parameters = [	
			{'n_estimators': [2**5, 2**6, 2**7, 2**8, 2**9], 'criterion': ['entropy']},
		]
		clf = model_selection.GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring=score, verbose=0, n_jobs=5)

	clf.fit(x_train, y_train)
	print(clf.best_params_)
	clf.best_estimator_.fit(x_train, y_train)
	return clf


def verification(train_set, train_meta, test_set, test_meta, feature, classifier, fusion):
	"""
	Performs the verification task and prints the results (confusion matrix, report, average and std of the normalized accuracies among users)

	Parameters
    ----------
		train_set: [M, `frame_size`, `n_accelerations`] `ndarray` the frames for training given that M is the total number of frames obtained from the training sessions
    	train_meta: [M, `dim_meta`] `ndarray` meta information about the training frames as user id of each frame
    	test_set: [L, `frame_size`, `n_accelerations`] `ndarray` the frames for test given that L is the total number of frames obtained from the test sessions
    	test_meta: [L, `dim_meta`] `ndarray` meta information about the test frames as user id of each frame 
    	feature: `string` function to calculate the features (options implemented on ``descriptors.py``: `resultant_vector`, `time`, `freq` and `wavelet`)
		classifier: `string` name of the classifier to be adopted (options implemented: `svm` and `rd_forest`)
		fusion: `string` type of temporal fusion to be used (options implemented on ``temporal_coherence.py``: `max_prob`, `mean` and `voting`)
    """
	n_samples_class_pos = 150
	rate_class_neg = 1
	fold = sys.argv[6]
		
	print("\n--------- VERIFICATION ---------")

	print("train set: " + str(train_set.shape))
	print("test set: " + str(test_set.shape))

	total_score_norm = []
	train_features, test_features = feature(train_set, test_set)

	classes = np.unique(train_meta[:, label_user])
	for c, klass in enumerate(classes):
		print("------------------------------\n* class : " + str(klass))
		x_train, user_train_meta = sampling(train_features, train_meta, klass, label_user, n_samples_class_pos, rate_class_neg, fold, path_idxs)
		y_train = user_train_meta[:, label_user]

		x_test = test_features
		y_test = test_meta[:, label_user]
		idxs_pos_test = np.where(y_test==klass)[0]
		y_test[:] = -1
		y_test[idxs_pos_test] = 1

		print('x_train:', x_train.shape)
		print('x_test:', x_test.shape)

		x_train, x_test = min_max(x_train, x_test)

		clf = trainModel(x_train, y_train, classifier, 'roc_auc')

		if(hasattr(temporal_coherence, fusion)):
			fusion_method = eval(fusion)
			probabilities = clf.best_estimator_.predict_proba(x_test)
			labels_sessions_test = test_meta[:, label_session]
			y_true, y_pred, prob_final = fusion_method(probabilities, y_test, labels_sessions_test)
		else:
			y_true, y_pred = y_test, clf.best_estimator_.predict(x_test)

		matrix = metrics.confusion_matrix(y_true, y_pred).T
		score_norm = np.average(metrics.recall_score(y_true, y_pred, average=None))
		total_score_norm.append(score_norm)

		print(metrics.classification_report(y_true, y_pred))
		print("----------------------------")
		print(matrix)
		print("----------------------------")
		print("acc norm " + str(score_norm))
		print("----------------------------")
		print("")

	print("Average score: ", np.mean(total_score_norm))
	print("Std: ", np.std(total_score_norm))



if __name__ == "__main__":
	subset = sys.argv[1]
	mode = eval(sys.argv[2])
	feature = eval(sys.argv[3])
	classifier = sys.argv[4]
	fusion = sys.argv[5]

	train_set, train_meta, test_set, test_meta = loadData(path, subset, train_session='session1', test_session='session2')
	mode(train_set, train_meta, test_set, test_meta, feature, classifier, fusion)


