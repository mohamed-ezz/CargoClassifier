helptxt="""This is the MAIN application for this project, it is the engine that concatenates the whole pipeline. It takes an image, or a directory containing
images, process images through the segmentation, then classify each found object, and shows the result visually
"""

import argparse
from kmeans import KmeansExtractor
import idputils
import cv2, cv, numpy as np
import pickle
from extraction.learner import Learner
import config
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=helptxt)
	parser.add_argument('-i', dest='colorImagePath', help="Path or Directory of color image(s) to process. The depth image should be located in the same directory.Eg: 15251260000_COLOR.bmp and 15251260000_DEPTH.png")
	parser.add_argument('-multiclf', default=None, metavar='PickledClassifierModelPath', dest='multiclf', help="Path to a pickled scikit multiclass classifier object with 0-3 output.")
	parser.add_argument('-feature', metavar='FeatureName', dest='feature', help='Feature transform to be applied to the image before classifying. One of hog, hue, gray.')
	parser.add_argument('-classifyonly', dest='classifyOnly',default=False, action='store_true', help='Skip segmentation, directly classify the given image.')
	parser.add_argument('-dontshow', dest='dontshow',default=False, action='store_true', help='Dont show image, just classify all and show total stats')
	parser.add_argument('-o', dest='output_path',default=None, help='Path of output csv file having object segmentation bounding boxes And object type. 0=Barrel,1=Blue,2=Brown,3=Irrelevant ')
	
	args = parser.parse_args()
	
	if args.classifyOnly:
		print 'Skipping segmentation, directly classifying with %s features.' % args.feature
	
	#Get list of images to process
	if os.path.isdir(args.colorImagePath):
		lst = idputils.list_images(args.colorImagePath)
		colornamelist = zip(*lst)[0] #get list of colornames
	else:
		colornamelist = [args.colorImagePath]
	
	#Now for each image
	total_barrels, total_blues, total_browns = 0,0,0
	total = len(colornamelist)
	extractor = KmeansExtractor()
	multiclf = pickle.load(open(args.multiclf))
	if args.output_path:
		csv_out = open(args.output_path, 'w')
		csv_out.write("image_id,box_id,y1,x1,y2,x2,objectlabel\n")
	for colorname in colornamelist:
		print colorname
		colorimage = cv2.imread(colorname)
		depthname = idputils.to_depth_filename(colorname)
		prefix    = idputils.get_filename_prefix(colorname)
		
		#SEGMENT AND GET OBJECT IMAGES
		if args.classifyOnly:
			#objectimgs = [colorimage]
			rects = [colorimage.shape]
		else:
			rects = extractor._segment(colorimage, depthname, prefix, write_output=False)
			rects = map(lambda rect: np.array(rect)/config.PREDICT_AT_SCALE, rects) #RESCALE
			[idputils.red_rect(colorimage, *rect) for rect in rects] #draw rects on image
			#objectimgs = idputils.rects_to_objectimages(colorimage, rects)
		
		#CLASSIFY
		
		learner = Learner(feature=args.feature)
		barrels, blues, browns = 0,0,0
		#for objectimg in objectimgs:
		objid=0
		for y1,x1,y2,x2 in rects:
			objid+=1
			objectimg = colorimage[y1:y2+1, x1:x2+1]
			vector   = learner._image_to_featurevector(objectimg)
			prediction = multiclf.predict(vector)
			barrels += 1 if prediction==0 else 0
			blues   += 1 if prediction==1 else 0
			browns  += 1 if prediction==2 else 0
			
			if args.output_path and prediction != 3: #don't write if object is predicted as "irrelevant"
				csv_out.write("%s,%i,%i,%i,%i,%i,%i\n" % (prefix, objid, y1,x1,y2,x2, prediction))
			
		
		total_barrels += barrels
		total_blues   += blues
		total_browns  += browns
		
		text = "%i Barrels, %i Blues, %i Browns" % (barrels, blues, browns)
		print text
		
		if not args.dontshow:
			cv2.putText(colorimage, text, (0,colorimage.shape[0]-23), cv.CV_FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
			cv2.putText(colorimage, text, (0,colorimage.shape[0]-20), cv.CV_FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
			cv2.putText(colorimage, text, (0,colorimage.shape[0]-18), cv.CV_FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
			idputils.imshow(colorimage, '!!')
	
	csv_out.close()	
	print 'total_barrels:%i'%total_barrels
	print 'total_blues:%i'%total_blues
	print 'total_browns:%i'%total_browns
	print 'total:%i'%total
	
	
	
	