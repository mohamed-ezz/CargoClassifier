import numpy as np
import cv2,cv
import os,sys
import idputils
import config as cfg
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle
from skimage.feature import hog
from sklearn.metrics import classification_report

class Learner:
	def __init__(self, feature='hog', dirs = None, test_portion = 0.2, image_size = (64,64), datapicklefiles=None):
		self.dirs = dirs
		self.test_portion = test_portion
		self.image_size = image_size
		self.feature = feature # 'hue' or 'hog' or 'gray'
		if feature not in ['hue', 'hog', 'gray']:
			raise ValueError("Invalid value for feature: %s. It should be one of: hog, hue, gray" % feature)
		
		self.clf = None
		self.data = 'None' #write as string, bcoz otherwise numpy array will be compared elementwise with value None (in a future numpy version).
		if datapicklefiles: #if data is ready in a pickle file
			self.read_data_matrix(datapicklefiles)
	
	def load_classifier(self, clfpicklepath):
		self.clf = pickle.load(open(clfpicklepath,'r'))
		
	def pickle_classifier(self, outputpath):
		if self.clf:
			pickle.dump(self.clf,open(outputpath,'w'))
		else:
			print 'pickle_classifier found no saved classifier to save. learner.clf=None'
		
	def pickle_data_matrix(self, outputpath):
		if self.data == 'None':
			raise Exception('data matrix is not yet read. Cannot save it to a pickle file.')
		
		directory = os.path.dirname(outputpath)
		if not os.path.exists(directory):
			os.makedirs(directory)
			
		pickle.dump(self.data, open(outputpath,'w'))
		
	def read_data_matrix(self, datapicklefiles=None):
		"""Read data from pickled file if given in datapicklefiles, or from images in self.dirs (each dir = a class)"""
		
		if datapicklefiles != None:
			data = pickle.load(open(datapicklefiles[0],'r'))
			remainingfiles = datapicklefiles[1:]
			for datafile in remainingfiles:
				data2 = pickle.load(open(datafile,'r'))
				data = np.concatenate((data, data2),1) #concatenate columns
			self.data = data
			print 'Matrix loaded from %s' % datapicklefiles
		if self.data != 'None':
			return self.data
		

		X, y = 'None', 'None'
		for lbl, d in enumerate(self.dirs):
			if not os.path.isdir(d): continue
			if X=='None' and y=='None': #first timer
				X, y = self._read_dir_to_matrix(d, lbl)
			else:
				Xi, yi = self._read_dir_to_matrix(d, lbl)
				X = np.concatenate((X, Xi))
				y = np.concatenate((y, yi))
			
		y = np.reshape(y,(y.shape[0],1)) #convert shape (4000,) to (4000, 1)
		self.data = np.concatenate((X,y),1)

		print 'Data ready in memory. Matrix size:%s' % (str(self.data.shape))
		if self.feature=='hog':
			self.data = self.data.astype('float32') #for some reason, without this line, the matrix ends up being float64.
		elif self.feature in ['hue', 'gray']:
			self.data = self.data.astype('uint8')
		
		
		# Divide into train+validation and test datasets
		nrows = int(0.7 * self.data.shape[0])
		self.test_data = self.data[nrows:,:]
		self.data = self.data[:nrows,:]
		
		return self.data
	
	def _read_dir_to_matrix(self, directory, label):
		"""Returns 2-tuple (data,y) corresponding to images in a single directory, all having the same label
		Where data is the X feature matrix. y is the corresponding class label vector (y is just a vector of same values = label)"""
		
		print 'getting size'
		if self.feature == 'hog':
			n_features = 1764
		else:
			n_features = self.image_size[0] * self.image_size[1]
		
		n_rows = len(idputils.list_images(directory))
		
		print 'reading data to memory'
		if self.feature == 'hog':
			data = np.zeros((n_rows, n_features), np.float32)
		else:
			data = np.zeros((n_rows, n_features), np.uint8)
			
		idx = 0
		imagenames = idputils.list_images(directory)
		for colorpath, depthpath, prefix in imagenames:	
			colorimage = cv2.imread(colorpath)
			#preprocessing + feature selection
			vector = self._image_to_featurevector(colorimage)
			data[idx,:] = vector
			idx += 1
		
		print 'X.shape=',data.shape
		y   = np.ones((n_rows)) * label #class label (initialize with -1)
		return data,y

	def _image_to_featurevector(self, colorimage):
		if self.feature == 'hue': return self._image_to_featurevector_HUE(colorimage)
		if self.feature == 'hog': return self._image_to_featurevector_HOG(colorimage)
		if self.feature == 'gray': return self._image_to_featurevector_GRAY(colorimage)
		
	def _image_to_featurevector_HUE(self, colorimage):
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
	
	def _image_to_featurevector_HOG(self, colorimage):
		"""Takes colorimage (numpy ndarray) and does :
		preprocessing ,feature extraction ,feature selection...etc
		Returns a vector (numpy 1D array)"""
		grayimage =  cv2.cvtColor(colorimage,cv.CV_BGR2GRAY)
		#Resize
		grayimage = cv2.resize(grayimage, (64,64))
		#Switch to vector
		vector = hog(grayimage,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualise=False)
		#print 'hog vector shape',vector.shape
		return vector
	
	def _image_to_featurevector_GRAY(self, colorimage):
		"""Takes colorimage (numpy ndarray) and does :
		preprocessing ,feature extraction ,feature selection...etc
		Returns a vector (numpy 1D array)"""
		grayimage =  cv2.cvtColor(colorimage,cv.CV_BGR2GRAY)
		#Resize
		grayimage = cv2.resize(grayimage, self.image_size)
		#Switch to vector
		vector = grayimage.reshape(grayimage.size)
		return vector

	def balanced_accuracy(self, y, y_pred):
		"""Calculates the classification accuracy for unbalanced classes. 
		The error is the average of errors for each indiviual class. Accuracy = 1 - error"""
		if y.shape[0] != y_pred.shape[0]:
			raise ValueError('weighted_error: given arrays have different lengths : %i and %i' % (y.shape[0],y_pred.shape[0]))

		errors = []
		classes = np.unique(y)
		for lbl in classes:
			classpredictions = y_pred[y==lbl]
			classsize = classpredictions.size
			misclassified = (classpredictions!=lbl).sum()
			errors.append(misclassified*1.0/classsize)
		return 1 - np.average(errors)
	
	def sample_balanced(self, Xy, classsize = None):
		"""Returns a sub/over sampled balanced dataset. 
		For ex. if classsize=500, then 500 instances of each class will be taken from self.data"""
		np.random.seed(583)
		y=Xy[:,-1]
		print y.shape
		classcount = classsize or min(np.bincount(y))
		classes = np.unique(y)
		Xy_sampled = np.zeros((0,Xy.shape[1]))
		for lbl in classes:
			Xyclass = Xy[y==lbl]
			sample_idx = np.random.random_integers(0, Xyclass.shape[0]-1, classcount)
			Xy_sampled = np.concatenate((Xy_sampled, Xyclass[sample_idx,:]))
		return Xy_sampled
		
	def test(self, X_test=None, y_test=None, frompickle=None):
		if frompickle:
			data = pickle.load(open(frompickle,'r'))
			X_test, y_test = data[:,:-1], data[:,-1]
		y_predicttest = self.clf.predict(X_test)
		test_accuracy  = self.balanced_accuracy(y_test, y_predicttest)
		print classification_report(y_test, y_predicttest, target_names=['Barrel','Blue','Brown','Non-object'])
		print confusion_matrix(y_test, y_predicttest)
		print 'Test accuracy: ', test_accuracy	
		return test_accuracy
	

		
	def train_test(self, clf, title = 'Untitled', test_size = 0.3, presample_class_size = None):
		
		data = self.data
		if presample_class_size: #subsample with balanced class proportions
			data = self.sample_balanced(self.data, presample_class_size)
			
		X,y = data[:,:-1], data[:,-1].astype(np.uint8) #separate last column to a vector y
		skfold = cross_validation.StratifiedShuffleSplit(y, n_iter=1, test_size = test_size, random_state = 583)
		for train_index, test_index in skfold:
			X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

		print '======================'
		print 'Xtrain shape ',X_train.shape,' --- Xtest', X_test.shape
		clf.fit(X_train,y_train)
		self.clf = clf
		y_predicttrain = clf.predict(X_train)
		y_predicttest = clf.predict(X_test)	
		train_accuracy = self.balanced_accuracy(y_train, y_predicttrain)
		test_accuracy  = self.balanced_accuracy(y_test, y_predicttest)
		
		print clf	
		print y_test.dtype,y_test
		print y_predicttest.dtype,y_predicttest
		print classification_report(y_test, y_predicttest, target_names=['Barrel','Blue','Brown','Non-object'])
		print confusion_matrix(y_test, y_predicttest)
		print 'Test:', test_accuracy, '    Train:', train_accuracy
		print '======================'
		return {'train_accuracy':train_accuracy, 
			    'test_accuracy':test_accuracy, 
			    'n_samples_train':X_train.shape[0], 
			    'n_samples_test':X_test.shape[0]}

# 	def train_test(self, clf, title = 'Untitled'):
# 		
# 		X,y = self.data[:,:-1], self.data[:,-1] #separate last column to a vector y
# 		#Learn and predict
# 		SEED = 583
# 		k = 1
# 		skfold = cross_validation.StratifiedShuffleSplit(y, n_iter=k, test_size = 0.3, random_state = SEED)
# 		avg_test_accuracy = 0
# 		avg_train_accuracy = 0
# 		for i, d in enumerate(skfold):
# 			train_index, test_index = d
# 			X_train,X_test = X[train_index], X[test_index]
# 			y_train,y_test = y[train_index], y[test_index]
# 
# 			clf.fit(X_train,y_train)
# 			y_predicttrain = clf.predict(X_train)
# 			accuracy = metrics.accuracy_score(y_train, y_predicttrain)#f1_score(y_train, y_predicttrain)
# 			avg_train_accuracy += accuracy
# 			#print '\t\t\t\tTrain accuracy: ', accuracy
# 			y_predicttest = clf.predict(X_test)
# 			
# 			self.clf = clf
# 			
# 			accuracy = metrics.metrics.accuracy_score(y_test, y_predicttest)#f1_score(y_test, y_predicttest)
# 			#idputils.plot_prec_recall([y_test], [y_predicttest[:,0]], ['svm'], 'TITLE', 'Pre_Rec_%i'%i)
# 			print confusion_matrix(y_test, y_predicttest)
# 			#print 'Test accuracy: ', accuracy
# 			avg_test_accuracy += accuracy
# 			sys.stdout.write('.');sys.stdout.flush()
# 			#print 'Parameters:',clf.coef_
# 			
# 		sys.stdout.write('\n')
# 		avg_test_accuracy /= k
# 		avg_train_accuracy /= k
# 		print title, 'Classifier'
# 		print 'Test:',avg_test_accuracy,'    Train:',avg_train_accuracy
# 		return y_predicttest

	def predictimg(self, img):
		"""Use self.clf to predict given img. img could be a path or a numpy array image"""
		if type(img) == str:
			img = cv2.imread(img)
		
		vector = self._image_to_featurevector(img)
		y = self.clf.predict(vector)
		print y
		return y[0]
	
	def predictdir(self, directory):
		
		lst = idputils.list_images(directory)
		y = []
		for colorname,_,_ in lst:
			y.append(self.predictimg(colorname))
		
		return np.array(y)
		
		
if __name__ == '__main__':
	
	"""
	python learner.py -matrix /labeled/SETS/06
	"""
	import argparse
	parser = argparse.ArgumentParser()
	#python learner.py -matrix /labeled/
	parser.add_argument('-matrix', metavar = 'path/file.pickle',nargs='+', dest='readMatrixPaths', help = 'Path to pickle file(s) with a numpy data matrix with last column having labels. If more than one file is given, they should have same # of instances, the features from both matrices will be concatenated and used together.')
	parser.add_argument('-trainsize', dest ='trainsize', default=0.1,type=float, help='Proportion of data to use for training, rest is for testing.Float between 0-1.')
	parser.add_argument('-d',dest='dirs', nargs='+', help='One or more directories with the images, each dir should contain one class to be classified.')
	parser.add_argument('-saveonly', metavar = 'path/file.pickle', dest='saveMatrixPath', help='Make the command only read the given images data, saves it to a matrix and Stop.')
	parser.add_argument('-feature', dest ='feature', default=None, help='The feature type to extract. Either hog, hue or gray')
	parser.add_argument('-savemodel', dest='modeloutputfile', default=None, help= 'Path to file to save the classifier model to. If this option is not used, the model will not be saved.')
	
	args = parser.parse_args()
	if args.readMatrixPaths and (args.dirs or args.saveMatrixPath or args.feature):
		parser.error("Cannot use -matrix with one of -feature,-d,-save-only")

	if not args.feature:
		args.feature='hog'
############################################################################################################################
	if  args.readMatrixPaths:
		learner = Learner(args.feature, datapicklefiles= args.readMatrixPaths)
	else:
		learner = Learner(args.feature, dirs = args.dirs)
		learner.read_data_matrix()
	
	from sklearn.svm import LinearSVC
	from sklearn.svm import SVC
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.neighbors import NearestCentroid
	from sklearn.ensemble import AdaBoostClassifier
	#from nolearn.dbn import DBN
	import nolearn
	from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
	from sklearn.cross_validation import train_test_split
	from collections import namedtuple
	
	
	ModelTune = namedtuple('ModelTune', 'model params')
	SEED=583
	classifiers = [
	#ModelTune(DecisionTreeClassifier(), {'max_depth':[9], 'class_weight':['auto'], 'random_state':[SEED]})
	#ModelTune(LinearSVC(), {'C':[0.0000001, 0.01, 0.0001], 'loss':['hinge'], 'class_weight':['auto']}),
	#ModelTune(SVC(), {'C':[0.000001, 0.01, 0.0001], 'kernel':['rbf'],'degree':[2], 'gamma':[5], 'class_weight':['auto'], 'tol':[0.01]}),
	ModelTune(LogisticRegression(), {'C':[10],'intercept_scaling':[100000],'class_weight':['auto'],'random_state':[SEED]}),
	#ModelTune(RandomForestClassifier(), {'n_jobs':[6],'n_estimators':[20],'class_weight':['auto'], 'min_samples_split':[160],'random_state':[SEED]}),
	#ModelTune(NearestCentroid(), {}),
	#ModelTune(AdaBoostClassifier(), {'base_estimator':[SVC(kernel='linear', C=0.001, class_weight='auto')],'random_state':[SEED],'algorithm':['SAMME']}),
	#ModelTune(DBN(), {'layer_sizes':[[-1, 20, -1]], 'output_act_funct':[nolearn.dbn.activationFunctions.Sigmoid()]}),
	]
	if args.saveMatrixPath:
		learner.pickle_data_matrix(args.saveMatrixPath)
	else:
		results = []
		for classifier in classifiers:
			#clf = RandomizedSearchCV(classifier.model, classifier.params,n_iter=1, n_jobs=7, cv=2, verbose=5, pre_dispatch='n_jobs')
			#print clf
			for k in classifier.params: classifier.params[k] = classifier.params[k][0]
			clf=classifier.model.__class__(**classifier.params)
			res = learner.train_test(clf,test_size=1-args.trainsize, presample_class_size=None)
			results.append((clf, res))
		if args.modeloutputfile:
			learner.pickle_classifier(args.modeloutputfile)
	
		

#For 64x64 images
# Test accuracy: 0.912198824923
#y_predicttest=learner.train_test(LinearSVC(C = 0.001))
# LinearSVC(loss='l1', C = 0.000005)

# Test accuracy : 0.91957374675
# LogisticRegression('l2',C = 0.0001)
	
