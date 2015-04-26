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
	parser.add_argument('-feature', metavar='FeatureName', dest='feature', help='Feature transform to be applied to the image before classifying. One of hog, hue, gray.')
	parser.add_argument('-classifyonly', dest='classifyOnly',default=False, action='store_true', help='Skip segmentation, directly classify the given image.')
	parser.add_argument('-dontshow', dest='dontshow',default=False, action='store_true', help='Dont show image, just classify all and show total stats')
	
	
	args = parser.parse_args()
	
	# GET OBJECT IMAGES (SEGMENTATION)
	extractor = KmeansExtractor()
	if os.path.isdir(args.colorImagePath):
		lst = idputils.list_images(args.colorImagePath)
		colornamelist = zip(*lst)[0] #get list of colornames
	else:
		colornamelist = [args.colorImagePath]
	
	total_barrels, total_blues, total_browns = 0,0,0
	total = len(colornamelist)
	for colorname in reversed(colornamelist):
		print colorname
		colorimage = cv2.imread(colorname)
		depthname = idputils.to_depth_filename(colorname)
		prefix    = idputils.get_filename_prefix(colorname)
		
		if args.classifyOnly:
			objectimgs = [colorimage]
		else:
			rects = extractor._segment(colorname, depthname, prefix, write_output=False)
			scaledrects = []
			for rect in rects:
				rect=np.array(rect)
				rect /= config.PREDICT_AT_SCALE
				scaledrects.append(rect)
			rects = scaledrects
			objectimgs = idputils.rects_to_objectimages(colorimage, rects)
		
		#for o in objectimgs:
		#	idputils.imshow(o)
		
		#CLASSIFY
		
		barrelclf = pickle.load(open(args.barrelclf))
		blueclf = pickle.load(open(args.blueclf))
		brownclf = pickle.load(open(args.brownclf))
		
		learner = Learner(feature=args.feature)
		barrels, blues, browns = 0,0,0
		for objectimg in objectimgs:
			vector   = learner._image_to_featurevector(objectimg)
			barrels += barrelclf.predict(vector)[0]
			blues   += blueclf.predict(vector)[0]
			browns  += brownclf.predict(vector)[0]
		
		total_barrels += barrels
		total_blues   += blues
		total_browns  += browns
		
		text = "%i Barrels, %i Blues, %i Browns" % (barrels, blues, browns)
		print text
		
		if not args.dontshow:
			cv2.putText(colorimage, text, (0,colorimage.shape[0]-20), cv.CV_FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
			idputils.imshow(colorimage, '!!')
		
	print 'total_barrels:%i'%total_barrels
	print 'total_blues:%i'%total_blues
	print 'total_browns:%i'%total_browns
	print 'total:%i'%total
	
	
	
	