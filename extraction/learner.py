import numpy as np
import cv2,cv
import os,sys
import idputils
import config as cfg
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import pickle
from operator import add
from skimage.feature import hog

class Learner:
	def __init__(self, feature='hog', positive_dirs = None, negative_dirs = None, test_portion = 0.2, image_size = (64,64), datapicklefile=None):
		self.positive_dirs = positive_dirs #positive classes directories
		self.negative_dirs = negative_dirs #irrelevant / non-object images
		self.test_portion = test_portion
		self.image_size = image_size
		self.feature = feature # 'hue' or 'hog' or 'gray'
		if feature not in ['hue', 'hog', 'gray']:
			raise ValueError("Invalid value for feature: %s. It should be one of: hog, hue, gray" % feature)
		
		self.clf = None
		self.data = 'None' #write as string, bcoz otherwise numpy array will be compared elementwise with value None (in a future numpy version).
		if datapicklefile: #if data is ready in a pickle file
			self.read_data_matrix(datapicklefile)
	
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
		
	def read_data_matrix(self, datapicklefile=None):
		"""Read data from pickled file if given in datapicklefile, or from images in self.positive_dirs & self.negative_dirs"""
		
		if datapicklefile != None:
			self.data = pickle.load(open(datapicklefile,'r'))
			print 'Matrix loaded from %s' % datapicklefile

		if self.data != 'None':
			return self.data
		
		Xpos, ypos = self._read_dir_to_matrix(self.positive_dirs, 1)
		Xneg, yneg = self._read_dir_to_matrix(self.negative_dirs, 0)
		
		X = np.concatenate((Xpos,Xneg))
		y = np.concatenate((ypos,yneg))
		print y.shape
		y = np.reshape(y,(y.shape[0],1)) #convert shape (4000,) to (4000, 1)
		print y.shape
		self.data = np.concatenate((X,y),1)

		print 'Data ready in memory. Matrix size:%s' % (str(self.data.shape))
		return self.data
	
	def _read_dir_to_matrix(self, dirs, label):
		"""Returns 2-tuple (data,y) corresponding to images in a single directory, all having the same label
		Where data is the X feature matrix. y is the corresponding class label vector (y is just a vector of same values = label)"""
		
		print 'getting size'
		if self.feature == 'hog':
			n_features = 1764
		else:
			n_features = self.image_size[0] * self.image_size[1]
		
		lens = map(lambda posdir: len(idputils.list_images(posdir)), dirs)
		n_rows = reduce(add, lens)
		#n_rows = len(idputils.list_images(dirs))
		
		print 'reading data to memory'
		if self.feature == 'hog':
			data = np.zeros((n_rows, n_features), np.float32)
		else:
			data = np.zeros((n_rows, n_features), np.uint8)
			
		y   = np.ones((n_rows)) * -1 #class label (initialize with -1)
		idx = 0
		for posdir in dirs:
			imagenames = idputils.list_images(posdir)
			for colorpath, depthpath, prefix in imagenames:	
				colorimage = cv2.imread(colorpath)
				#preprocessing + feature selection
				#print 'Reading: ',colorpath
				vector = self._image_to_featurevector(colorimage)
				y[idx] = label
				data[idx,:] = vector
				#print idx
				idx += 1
		
		print idx
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
	
	def train_test(self, clf, title = 'Untitled'):
		
		X,y = self.data[:,:-1], self.data[:,-1] #separate last column to a vector y
		#Learn and predict
		SEED = 583
		k = 1
		skfold = cross_validation.StratifiedShuffleSplit(y, n_iter=k, test_size = 0.3, random_state = SEED)
		avg_test_accuracy = 0
		avg_train_accuracy = 0
		for i, d in enumerate(skfold):
			train_index, test_index = d
			X_train,X_test = X[train_index], X[test_index]
			y_train,y_test = y[train_index], y[test_index]
			
			clf.fit(X_train,y_train)
			y_predicttrain = clf.predict(X_train)
			accuracy = metrics.accuracy_score(y_train, y_predicttrain)#f1_score(y_train, y_predicttrain)
			avg_train_accuracy += accuracy
			#print '\t\t\t\tTrain accuracy: ', accuracy
			y_predicttest = clf.predict(X_test)
			
			self.clf = clf
			
			accuracy = metrics.metrics.accuracy_score(y_test, y_predicttest)#f1_score(y_test, y_predicttest)
			#idputils.plot_prec_recall([y_test], [y_predicttest[:,0]], ['svm'], 'TITLE', 'Pre_Rec_%i'%i)
			print confusion_matrix(y_test, y_predicttest)
			#print 'Test accuracy: ', accuracy
			avg_test_accuracy += accuracy
			sys.stdout.write('.');sys.stdout.flush()
			#print 'Parameters:',clf.coef_
			
		sys.stdout.write('\n')
		avg_test_accuracy /= k
		avg_train_accuracy /= k
		print title, 'Classifier'
		print 'Test:',avg_test_accuracy,'    Train:',avg_train_accuracy
		return y_predicttest

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
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-matrix', metavar = 'path/file.pickle', dest='readMatrixPath', help = 'Path to pickle file with a numpy data matrix with last column having labels')
	parser.add_argument('-p',dest='positive_dirs', nargs='+', help='One or more directories with the positive images.')
	parser.add_argument('-n', dest='negative_dirs', nargs='+')
	parser.add_argument('-saveonly', metavar = 'path/file.pickle', dest='saveMatrixPath', help='Make the command only read the given images data (with -p and -n), saves it to a matrix and Stop.')
	parser.add_argument('-feature', dest ='feature', default=None, help='The feature type to extract. Either hog, hue or gray')
	parser.add_argument('-savemodel', dest='modeloutputfile', default=None, help= 'Path to file to save the classifier model to. If this option is not used, the model will not be saved.')
	
	args = parser.parse_args()
	if args.readMatrixPath and (args.positive_dirs or args.negative_dirs or args.saveMatrixPath or args.feature):
		parser.error("Cannot use --matrix with one of -p,-n,--save-only")

	if not args.feature:
		args.feature='hog'
############################################################################################################################
	print args.feature
	if  args.readMatrixPath:
		learner = Learner(args.feature, datapicklefile= args.readMatrixPath)
	else:
		learner = Learner(args.feature, positive_dirs = args.positive_dirs, negative_dirs = args.negative_dirs)
		learner.read_data_matrix()
	
	
	if args.saveMatrixPath:
		learner.pickle_data_matrix(args.saveMatrixPath)
	else:
		#while True:
		#	c = float(input('Select a value for C: '))
		pass#y_predicttest=learner.train_test(SVC(kernel='linear',  C = 0.001, class_weight='auto'))
		if args.modeloutputfile:
			learner.pickle_classifier(args.modeloutputfile)
	
	
#For 64x64 images
# Test accuracy: 0.912198824923
#y_predicttest=learner.train_test(LinearSVC(loss='l2', C = 0.001))
# LinearSVC(loss='l1', C = 0.000005)

# Test accuracy : 0.91957374675
# LogisticRegression('l2',C = 0.0001)
	
