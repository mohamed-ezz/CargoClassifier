import numpy as np
import cv2,cv
import os,sys
import idputils
import config as cfg
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import metrics
from sklearn.svm import LinearSVC

class Learner:
	def __init__(self, dirs, negative_dir, test_portion = 0.2, image_size = (64,64)):
		self.image_dirs = dirs #positive classes directories
		self.negative_dir = negative_dir #irrelevant / non-object images
		self.test_portion = test_portion
		self.image_size = image_size
	
	def _read_data_matrix(self):
		"""Returns 2-tuple (data,y)
		Where data is the X feature matrix. y is the corresponding class label vector"""
		
		print 'getting size'
		n_rows = 0
		n_features = self.image_size[0] * self.image_size[1]
		for directory in self.image_dirs:
			n_rows += len(idputils.list_images(directory))
		
		print 'filling data'
		data = np.zeros((n_rows, n_features), np.uint8)
		y    = np.ones((n_rows)) * -1 #class label (initialize with -1)
		idx = 0
		for label, directory in enumerate(self.image_dirs):
			imagenames = idputils.list_images(directory)
			for colorpath, depthpath, prefix in imagenames:	
				colorimage = cv2.imread(colorpath)
				depthimage = cv2.imread(depthpath,-1)	
				#preprocessing + feature selection
				#print 'Reading: ',colorpath
				vector = self._image_to_featurevector(colorimage, depthimage)
				y[idx] = label
				data[idx,:] = vector
				#print idx
				idx += 1
		
		print idx
		return data,y

	def _image_to_featurevector(self, colorimage, depthimage):
		"""Takes colorimage (numpy ndarray) and does :
		preprocessing ,feature extraction ,feature selection...etc
		Returns a vector (numpy 1D array)"""
		#Threshold depth
# 		colorimage[depthimage < cfg.DEPTH_LO] = 0
# 		colorimage[depthimage > cfg.DEPTH_HI] = 0
		#Get hueimage only
		hueimage =  cv2.cvtColor(colorimage,cv.CV_BGR2HSV)[:,:,0]
		#Median filter to remove noise
		#hueimage = cv2.medianBlur(hueimage, 21) #found to worsen performance
		#Resize
		hueimage = cv2.resize(hueimage, self.image_size)
		#Switch to vector
		vector = hueimage.reshape((hueimage.size))
		return vector
	
	
	
	def learn_evaluate(self, X, y, clf):
		
		#Read data from directories..
		#X, y = self._read_data_matrix()
		#Learn and predict
		SEED = 583
		k = 10
		skfold = cross_validation.StratifiedKFold(y, n_folds=k, random_state = SEED)
		avg_test_accuracy = 0
		avg_train_accuracy = 0
		for train_index, test_index in skfold:
			X_train,X_test = X[train_index], X[test_index]
			y_train,y_test = y[train_index], y[test_index]
			
			clf.fit(X_train,y_train)
			y_predicttrain = clf.predict(X_train)
			accuracy = metrics.accuracy_score(y_train, y_predicttrain)
			avg_train_accuracy += accuracy
			#print '\t\t\t\tTrain accuracy: ', accuracy
			y_predicttest = clf.predict(X_test)
			accuracy = metrics.accuracy_score(y_test, y_predicttest)
			#print 'Test accuracy: ', accuracy
			avg_test_accuracy += accuracy
			sys.stdout.write('.');sys.stdout.flush()
			#print 'Parameters:',clf.coef_
			
		sys.stdout.write('\n')
		avg_test_accuracy /= k
		avg_train_accuracy /= k
		print 'Test:',avg_test_accuracy,'    Train:',avg_train_accuracy
		return y_predicttest
	
	
if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-p',dest='positive_dirs', nargs='3'
					, help='A space separated list of directories, each directory has images of a one of Positive the classes.'
					, default=['/idpdata/frontal_labeledobjects/BARREL/','/idpdata/frontal_labeledobjects/BLUEBOX/','/idpdata/frontal_labeledobjects/BROWNBOX/']
					)
	parser.add_argument('-n', dest='negative_dir', default = '/idpdata/frontal_labeledobjects/IRRELEVANT')
	args = parser.parse_args()
	
	learner = Learner(args.positive_dirs, args.negative_dir)
	#Read data from directories..
	#BARREL=0, BLUE=1, BROWN=2
	X, y = learner._read_data_matrix()
	#print np.sum(y==0), np.sum(y==1), np.sum(y==2)
	#X = X[np.logical_or(y==1 ,y==0)]
	#y = y[np.logical_or(y==1 ,y==0)]
	print y
	while True:
		c = float(input('Select a value for C: '))
		y_predicttest=learner.learn_evaluate(X, y, LogisticRegression('l2',C=c))
	
	
#For 64x64 images
# Test accuracy: 0.912198824923
# LinearSVC(loss='l1', C = 0.000005)

# Test accuracy : 0.91957374675
# LogisticRegression('l2',C = 0.0001)
	
	
	
	
	