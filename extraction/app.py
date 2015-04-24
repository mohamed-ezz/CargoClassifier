import argparse
from kmeans import KmeansExtractor
import idputils
import cv2, cv, numpy as np
import pickle
from extraction.learner import Learner
import config
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', dest='colorImagePath', help="Path or Directory of color image(s) to process. The depth image should be located in the same directory.Eg: 15251260000_COLOR.bmp and 15251260000_DEPTH.png")
	parser.add_argument('-barrelclf', metavar='PickledClassifierModelPath', dest='barrelclf', help="Path to a pickled scikit classifier object for Barrels.")
	parser.add_argument('-blueclf', metavar='PickledClassifierModelPath', dest='blueclf', help="Path to a pickled scikit classifier object for Blue boxes.")
	parser.add_argument('-brownclf', metavar='PickledClassifierModelPath', dest='brownclf', help="Path to a pickled scikit classifier object for Brown boxes.")
	args = parser.parse_args()
	
	# GET OBJECT IMAGES (SEGMENTATION)
	extractor = KmeansExtractor()
	if os.path.isdir(args.colorImagePath):
		lst = idputils.list_images(args.colorImagePath)
		colornamelist = zip(*lst)[0] #get list of colornames
	else:
		
		colornamelist = [args.colorImagePath]
	
	for colorname in colornamelist:
		print colorname
		colorimage = cv2.imread(colorname)
		depthname = idputils.to_depth_filename(colorname)
		prefix    = idputils.get_filename_prefix(colorname)
		rects = extractor._segment(colorname, depthname, prefix, write_output=False)
		scaledrects = []
		for rect in rects:
			rect=np.array(rect)
			rect /= config.PREDICT_AT_SCALE
			scaledrects.append(rect)
		rects = scaledrects
		objectimgs = idputils.rects_to_objectimages(colorimage, rects)
		
		for o in objectimgs:
			idputils.imshow(o)
		
		#CLASSIFY
		barrelclf = pickle.load(open(args.barrelclf))
		blueclf = pickle.load(open(args.blueclf))
		brownclf = pickle.load(open(args.brownclf))
		
		learner = Learner()
		barrels, blues, browns = 0,0,0
		for objectimg in objectimgs:
			vector   = learner._image_to_featurevector(objectimg, feature='hog')
			barrels += barrelclf.predict(vector)[0]
			blues   += blueclf.predict(vector)[0]
			browns  += brownclf.predict(vector)[0]
			
		text = "%i Barrels, %i Blues, %i Browns" % (barrels, blues, browns)
		cv2.putText(colorimage, text, (0,colorimage.shape[0]-20), cv.CV_FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
		idputils.imshow(colorimage, '!!')
	
	
	