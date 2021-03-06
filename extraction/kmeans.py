from extraction.objectExtract import ObjectExtractor
import cv2,cv
import numpy as np
import sklearn.preprocessing
from sklearn.cluster import KMeans
import idputils
from scipy import stats
from extraction import config as cfg
import sys

#dict storing deltas of: [y1 x1 y2 x2], used in region growing step
GROW_DIRECTION = {
		    #                        y1,x1,y2,x2
			"GROW_LEFT":   np.array([ 0,-1, 0, 0]),
			"GROW_RIGHT":  np.array([ 0, 0, 0, 1]),
			"GROW_UP":     np.array([-1, 0, 0, 0]),
			"GROW_DOWN":   np.array([ 0, 0, 1, 0]),
			
			"SHRINK_LEFT": np.array([ 0, 0, 0,-1]), #the right side of the rect shrinks to the left
			"SHRINK_RIGHT":np.array([ 0, 1, 0, 0]), #left side shrink right
			"SHRINK_UP":   np.array([ 0, 0,-1, 0]), #bottom side move up
			"SHRINK_DOWN": np.array([ 1, 0, 0, 0]), #top side move down
			}
STEP_SIZE = 3
	
class KmeansExtractor(ObjectExtractor):

	
	def _region_grow_score(self, image):
		"""Returns a best_score for an image region. The best_score is maximum for regions that are :
		1-Big in area
		2-All have same color (more or less)
		3-Contains minimum number of 0 pixels (in our case, they mean background pixels)
		Image is expected to be 3 channel image after kmeans discretization.
		However all 3 channels have the same value (yes, redundant and needs to be fixed)
		""" 
		if len(image.shape) > 2 and image.shape[2] > 1:
		#Take 1 channel only of the given image if it has a 3rd dimension (channel)
			image = image[:,:,0]
		
#		Calculate Entropy in bits (for Non-bg pixels only) (the bigger the worse)
#		=============
		bins_all = np.bincount(np.reshape(image,(image.shape[0]*image.shape[1]))) #at each index i, value= (count of occurrences of i)
		unique = bins_all.nonzero()[0] #indices of nonzero, which = unique values in image 
		counts = bins_all[unique]
		counts = counts[1:] #ignore count of bg pixels
		inv_entropy = 1.0/(stats.entropy(counts,base=2)+1.0) #inverse of Entropy in bits (regularized with +1)
		
		area_percent = image.size*1.0 / (640*480*cfg.PREDICT_AT_SCALE**2)
		
		non_bg_percent= (image != 0).sum()*1.0 / (image.size+1)
		
		w_area, w_nonbg, w_inventropy = cfg.AREA_WEIGHT, cfg.NONBG_WEIGHT, cfg.INVENTROPY_WEIGHT
		best_score= w_area*area_percent + w_nonbg*non_bg_percent + w_inventropy*inv_entropy
		
		#AD HOC RULES
		if image.shape[1] / (cfg.PREDICT_AT_SCALE*640) > 0.25: return 0
		
		return best_score
# 		===========================================
		
		
#		Using discrete "unique count"-based variance
# 		variance = np.unique(image).shape[0]**6
# 		bg_pixels_count = (image==0).sum()**1.15
# 		area = image.size
# 		print 'area,%s' % str(area)
# 		print 'variance,%s' % str(variance)
# 		print 'bg_pixels_count,%s' %str(bg_pixels_count)
# 		return area - variance - bg_pixels_count
	
	def _find_least_variance_box(self, image):
		""" Iterates over all possible rectangle boundaries/roi's and draws the best. Takes a 1 channel image"""
		height = image.shape[0]
		width = image.shape[1]
		
		########################
		####### FIND SEED ######
		########################
		wsize = 10
		hsize = 10
		step = 15
		max_seed_score = float('-inf')
		best_rect = None #np.array([y1,x1,y2,x2])
		for y1 in xrange(0,height-hsize,step):
			for x1 in xrange(0,width-wsize,step):
				for config in xrange(1,2):
					y2 = int(y1+ config* cfg.SEED_HEIGHT * cfg.PREDICT_AT_SCALE)
					x2 = int(x1+ config* cfg.SEED_WIDTH  * cfg.PREDICT_AT_SCALE)
					best_score = self._region_grow_score(image[y1:y2+1, x1:x2+1]) 
					if best_score > max_seed_score:
						max_seed_score = best_score
						best_rect = np.array([y1,x1,y2,x2])
		##################################
		####### REGION GROWING ###########
		##################################
		def crop(image, rect):
			y1,x1,y2,x2 = rect #tuple
			return image[y1:y2+1, x1:x2+1]
		def grow(rect, full_image, DIRECTIONS, old_score):
			"""Modifies rect in place by growing it in the specified direction (string)
			according to best_score on the given image.
			Returns new best_score if growing improves old_score, or None if growing did not take place because it 
			worsens the best_score"""
			DIRECTION = np.zeros(4)
			for dire in DIRECTIONS:
				DIRECTION += GROW_DIRECTION[dire]
				
			new_rect3= np.array(rect) + STEP_SIZE*3 * DIRECTION #grow/shrink rect
			new_rect2= np.array(rect) + STEP_SIZE*2 * DIRECTION #grow/shrink rect
			new_rect1= np.array(rect) + STEP_SIZE*1 * DIRECTION #grow/shrink rect
			new_rect4= np.array(rect) + STEP_SIZE*4 * DIRECTION #grow/shrink rect
			rects = [ new_rect1, new_rect2, new_rect3, new_rect4]
			scores= map(lambda rect: self._region_grow_score(crop(full_image,rect)), rects)
			max_idx = np.argmax(scores)
			best_rect = rects[max_idx]
			new_score = scores[max_idx]
			if new_score > old_score: #assess new best_score
				rect[0]=best_rect[0]; rect[1]=best_rect[1]; rect[2]=best_rect[2]; rect[3]=best_rect[3] #modify in-place
				return new_score
			else:
				return None
			

# REGION GROWTH ALGO 3 (GROW SHRINK ANY ORDER):	
# 		best_score = float('-inf')
# 		prev_iteration_score = None
# 		
# 		while prev_iteration_score != best_score:
# 			prev_iteration_score = best_score
# 			best_score = grow(best_rect, image, ["GROW_LEFT"], best_score) or best_score #return 1st non-null value
# 			best_score = grow(best_rect, image, ["GROW_RIGHT"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["GROW_DOWN"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["GROW_UP"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["SHRINK_LEFT"], best_score) or best_score #return 1st non-null value
# 			best_score = grow(best_rect, image, ["SHRINK_RIGHT"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["SHRINK_DOWN"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["SHRINK_UP"], best_score) or best_score			
# 		y1,x1,y2,x2 = best_rect #unpack np array
# 		return y1,x1,y2,x2, best_score
############################################################



##########################

# REGION GROWTH ALGO 2 (GROW ANY ORDER):
		best_score = float('-inf')
		prev_iteration_score = None
		
		while prev_iteration_score != best_score:
			prev_iteration_score = best_score
			best_score = grow(best_rect, image, ["GROW_LEFT"], best_score) or best_score #return 1st non-null value
			best_score = grow(best_rect, image, ["GROW_RIGHT"], best_score) or best_score
			best_score = grow(best_rect, image, ["GROW_DOWN"], best_score) or best_score
			best_score = grow(best_rect, image, ["GROW_UP"], best_score) or best_score
			

# 			best_score = grow(best_rect, image, ["GROW_LEFT","SHRINK_UP"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["GROW_LEFT","SHRINK_DOWN"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["GROW_RIGHT","SHRINK_UP"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["GROW_RIGHT","SHRINK_DOWN"], best_score) or best_score	
# 			best_score = grow(best_rect, image, ["GROW_UP","SHRINK_LEFT"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["GROW_UP","SHRINK_RIGHT"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["GROW_DOWN","SHRINK_LEFT"], best_score) or best_score
# 			best_score = grow(best_rect, image, ["GROW_DOWN","SHRINK_RIGHT"], best_score) or best_score	
			
		y1,x1,y2,x2 = best_rect #unpack np array
		return y1,x1,y2,x2, best_score
##########################


# REGION GROWTH ALGO 1 : left,left, left. Right Right. Down,Down,Down. Up Up. End
# 		best_score = max_seed_score
# 		prev_iteration_score = None
# 		print best_rect
# 		while True:
# 			prev_iteration_score = best_score
# 			best_score = grow(best_rect, image, "GROW_LEFT", best_score) or best_score #return 1st non-null value
# 			if prev_iteration_score == best_score: #new best_score is NOT different
# 				break
# 		print 'LEFT done'
# 		print best_rect
# 		while True:
# 			prev_iteration_score = best_score
# 			best_score = grow(best_rect, image, "GROW_RIGHT", best_score) or best_score #return 1st non-null value
# 			if prev_iteration_score == best_score: #new best_score is NOT different
# 				break
# 		print 'RIGHT done'
# 		print best_rect
# 		while True:
# 			prev_iteration_score = best_score
# 			best_score = grow(best_rect, image, "GROW_DOWN", best_score) or best_score #return 1st non-null value
# 			if prev_iteration_score == best_score: #new best_score is NOT different
# 				break
# 		print 'DOWN done'
# 		print best_rect	
# 		while True:
# 			prev_iteration_score = best_score
# 			best_score = grow(best_rect, image, "GROW_UP", best_score) or best_score #return 1st non-null value
# 			if prev_iteration_score == best_score: #new best_score is NOT different
# 				break			
# 		print 'UP done'
# 		print best_rect
# 		
# 		y1,x1,y2,x2 = best_rect #unpack np array
# 		return y1,x1,y2,x2, best_score
##############################################################################
	
	
	def _toKmeansMatrix(self, colorimage,depthimage, n_fgpixels):
		"""Converts colorimage and depth image, into a Kmeans-ready matrix of shape (n_samples,n_features)
		where n_samples=number of pixels, and n_features=3 color + 3 location=6
		colorimage is RGB image.
		n_fgpixels : number of foreground values (in desired depth range). This can be derived from images, but will be redundant step so we pass it here."""
		hsvimage = cv2.cvtColor(colorimage,cv.CV_BGR2HSV)
		height = hsvimage.shape[0]
		width = hsvimage.shape[1]
	
		#Count number of pixels in the desired depth range, to know the 
		#size of matrix to allocate
		#@TODO: SLOW TRICK. 
		#pixels = np.ones((height,width),dtype=np.uint8)
		#pixels[depthimage < self.depth_lo] = 0
		#pixels[depthimage > self.depth_hi] = 0
		#n_points = int(pixels.sum()) # number of non-back/foreground pixels. We want to cluster only those
# 		print 'n_points',n_points
		#Create data matrix
		data = np.zeros((n_fgpixels,4))
		index = 0
		#fill it
		for r in range(height):
			for c in range(width):
				#Location features
				if depthimage[r][c]<self.depth_lo or depthimage[r][c]>self.depth_hi:
					continue
				else:
					data[index][0]=r
					data[index][1]=c
					data[index][2]=depthimage[r][c]
					#Color features (HSV)
					data[index][3]=hsvimage[r][c][0] #pick hue only
					index += 1
	
		#now data is a matrix, let's standardize it
		return data

	def _do_clustering(self,data, n_clusters):
# 		print 'Start Kmeans'
		model=KMeans(n_clusters=n_clusters, n_init=8, random_state=982)
		data = sklearn.preprocessing.scale(data) #returns a copy!
		#data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)
		#PLOT 3D
		
# 		colors = map(str,list(sklearn.preprocessing.MinMaxScaler().fit_transform(data[:,3])))
# 		idputils.scatter3d(data[:,1],data[:,2],data[:,0],sample_percentage=0.02, labels=['X coordinate','DEPTH','HUE COLOR'],colors=colors)
		
		
		data[:,0] *= cfg.Y_KMEANS_SCALE
		data[:,1] *= cfg.X_KMEANS_SCALE
		data[:,2] *= cfg.DEPTH_KMEANS_SCALE #depth importance
		data[:,3] *= cfg.COLOR_KMEANS_SCALE #color importance
		
		labels = model.fit_predict(data)
# 		print 'Done Kmeans'
		return labels, data
	
		
	def _segment(self, colorname, depthname, prefix, output_clustered_image = False, write_output = True):	
		"""
		output_clustered_image: Will stop function after clustering and returns the resulting image (without fitting boxes)
		write_output : if true, the result of kmeans clustering will be written as a grayscale image, and also the end result will be written
						, which is the color image, with red rectangles drawn around the detected objects
		"""	
		colorimage = cv2.imread(colorname) if type(colorname)==str else colorname
		depthimage = cv2.imread(depthname,-1) if type(depthname)==str else depthname
		scaled_size = (int(640*cfg.PREDICT_AT_SCALE), int(480*cfg.PREDICT_AT_SCALE))
		colorimage, depthimage = cv2.resize(colorimage, scaled_size) , cv2.resize(depthimage, scaled_size)
		colorimage,depthimage, n_fgpixels = self._threshold_depth(colorimage, depthimage)
		height = colorimage.shape[0]
		
		#data is n_pixels x n_features matrix. Contains only foreground pixels.
		#Each row in data matrix is (y, x, z, hue)
		data = self._toKmeansMatrix(colorimage, depthimage, n_fgpixels) 
		
		
		labels,_ = self._do_clustering(data, n_clusters=4)
		#distrib = np.bincount(labels)/(1.0*labels.size)
		#print distrib
		#Put segmentation result onto image
		clustered_image = colorimage.copy() #image that will carry cluster numbers
		#label_to_color = {0:(9,255,255),4:(4,32,0),6:(6,172,57),3:(3,217,29),1:(1,179,140),5:(5,184,230),2:(2,98,217),7:(7,0,170),8:(8,115,191)}
		
		# SCATTER PLOT OF DATA MATRIX
		if False:
			colors = 200*np.ones((data.shape[0],1,3), dtype=np.uint8);colors[:,0,0] = data[:,3];
			colors = cv2.cvtColor(colors, cv.CV_HSV2RGB)[:,0,:]/255.
			idputils.scatter3d(data[:,1],data[:,0],data[:,2],sample_percentage=0.04, labels=['X','Y','Z'],colors=colors)	
			exit(0)
		
		for i in range(data.shape[0]):
			grayvalue = int((255.0/5)*(labels[i]+1))
			clustered_image[data[i][0], data[i][1]] = (grayvalue, grayvalue, grayvalue)
			#clustered_image[data[i][0], data[i][1]] = label_to_color[labels[i]]
		
		kernelsize = int(cfg.MEDIAN_BLUR_KERNEL_SIZE*cfg.PREDICT_AT_SCALE)
		clustered_image = cv2.medianBlur(clustered_image, kernelsize if kernelsize%2==1 else kernelsize+1)
		if output_clustered_image:
			return clustered_image
		
		if write_output: #segmentimage_orig will be used later only if write_output is True
			segmentimage_orig = clustered_image.copy() #save clustered image b4 it gets destroyed

		n_objects = cfg.N_OBJECTS
		rects = []
		for i in range(n_objects):
			y1,x1,y2,x2, regionscore = self._find_least_variance_box(clustered_image)
			if  regionscore > cfg.MIN_ACCEPTED_OBJECT_SCORE and y1*1.0/height < 0.9: #accept object if it has min score and appears/starts at top 80% 
				rects.append((y1,x1,y2,x2))
			clustered_image[y1:y2+1, x1:x2+1] = 0
			
	
		if write_output:
			for y1,x1,y2,x2 in rects:
				w = x2-x1; h=y2-y1;
				#print w,'x',h
				idputils.red_rect(colorimage, y1,x1,y2,x2)
				#idputils.red_rect(segmentimage_orig, y1,x1,y2,x2)
				pass
			
			if True:
				#segmentimage_orig = cv2.resize(segmentimage_orig, (640,480))
				segmentimage_orig[:,:,1:]= 255
				segmentimage_orig = cv2.cvtColor(segmentimage_orig, cv.CV_HSV2BGR) 
				idputils.imshow(colorimage,'color segment result')
				idputils.imshow(segmentimage_orig,'segment result')		
				#self.saveImage(prefix+'_COLOR.bmp', colorimage)
				#self.saveImage(prefix+'_SEGMENTS.png', segmentimage_orig)
		
		return rects
		
		
		
		
		
		
		
		
		
if __name__ == '__main__':
	import segmentation_error
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-i',dest='input_dir',  help='Input directory', default = '/idpdata/frontal/')
	parser.add_argument('-o',dest='output_dir', help="Output directory", default='/idpdata/seg_output/kmeans_growth_may19/')
	parser.add_argument('-l',dest='labeled_csv',help="Path to Labeled segmentation csv",default='/idpdata/frontal/segmentation_labels.csv')
	args = parser.parse_args()	



##### SINGLE RUN ON ALL IMAGES
	extractor = KmeansExtractor(args.input_dir, args.output_dir)
	#extractor = AggCluster(args.input_dir, args.output_dir)
	predicted_csv = extractor.extractAllObjects() #filepath+name
	if args.labeled_csv:
		print 'Comparing results with labeles.'
		best_score, fp, fn = segmentation_error.get_file_accuracy(args.labeled_csv, predicted_csv, cfg.PREDICT_AT_SCALE)

	exit(0)




#############
	import math
	def evaluate(weight_vector):
		print 'Evaluating:', weight_vector
		#cfg.Y_KMEANS_SCALE, cfg.X_KMEANS_SCALE, cfg.DEPTH_KMEANS_SCALE, cfg.COLOR_KMEANS_SCALE = weight_vector
		cfg.AREA_WEIGHT, cfg.NONBG_WEIGHT, cfg.INVENTROPY_WEIGHT = weight_vector
		extractor = KmeansExtractor(args.input_dir, args.output_dir)
		predicted_csv = extractor.extractAllObjects() 
		best_score, fp, fn = segmentation_error.get_file_accuracy(args.labeled_csv, predicted_csv, cfg.PREDICT_AT_SCALE)
		return best_score

	
	inits = []
	best_weights_for_init = []
	best_score_for_init = []

	while True: #for each initialization
		init=np.random.randn(3)
		init = init*np.sign(init)*10
		print 'Initializing to',init
		best_weights = np.array(init) #best_ = current _
		best_score = evaluate(best_weights)
		best_init = init
		
		improved = True
		while improved: #if false:stop when no improvement happened in all directions (continues to next init in the outer loop)
			for direction in [[1,0,0] ,[0,1,0], [0,0,1]]:
				direction = np.array(direction)
				new_weights = best_weights + 0.3 * direction
				new_score = evaluate(new_weights)
				sys.stdout.write('#');sys.stdout.flush()
				if new_score > best_score:
					improved=True
					best_score = new_score
					best_weights = new_weights
					best_init = init
					print 'New best score:',best_score,'at weights:',best_weights
					continue #No need to try -ve sign direction
				else:
					improved = False
				
				#Try direction with -ve sign
				new_weights = best_weights - 0.3 * direction
				new_score = evaluate(new_weights)
				sys.stdout.write('#');sys.stdout.flush()
				if new_score > best_score:
					improved=True
					best_score = new_score
					best_weights = new_weights
					best_init = init
					print 'New best score:',best_score,'at weights:',best_weights
				else:
					improved=False
		
		
		best_weights_for_init.append(best_weights)
		best_score_for_init.append(best_score)
		inits.append(init)
		best_idx = np.argmax(best_score_for_init)
		print 'Best Weights Vector : ', best_weights_for_init[best_idx]
		print 'Best Score : ', best_score_for_init[best_idx]
		print 'Best Initialization:', inits[best_idx]
		print '============================================='
		print '============================================='
		print '\n\n'
			
#############





#### GRID SEARCH TUNING

#### TUNE SCORE TERM WEIGHTS
# 	best_weights = (-1,-1,-1)
# 	best_score = float('-inf')
# 	one_ten = [1,2,3,4,5,6,7,8,9,10]
# 	for w_area in one_ten:
# 		for w_nonbg in one_ten:
# 			for w_inventropy in one_ten:
# 				cfg.AREA_WEIGHT=w_area
# 				cfg.NONBG_WEIGHT = w_nonbg
# 				cfg.INVENTROPY_WEIGHT = w_inventropy
# 				
# 				#Run segmentation	
# 				extractor = KmeansExtractor(args.input_dir, args.output_dir)
# 				#extractor = AggCluster(args.input_dir, args.output_dir)
# 				predicted_csv = extractor.extractAllObjects() #filepath+name
# 				
# 				
# 				#Compare result with labeled file.
# 				if args.labeled_csv:
# 					print 'Comparing results with labeles.'
# 					best_score, fp, fn = segmentation_error.get_file_accuracy(args.labeled_csv, predicted_csv, cfg.PREDICT_AT_SCALE)
# 					if best_score > best_score:
# 						best_weights = (w_area, w_nonbg, w_inventropy)
# 						best_score = best_score
# 					print '==============='
# 					print 'best_score,fp,fn : ',best_score,fp,fn
# 					print 'w_area, w_nonbg, w_inventropy : ', w_area, w_nonbg, w_inventropy
# 					print '==============='
# 
# 		
# 	print '=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-'
# 	print 'BEST SCORE : ',best_score
# 	print 'BEST WEIGHTS : ', best_weights
# 		
		
	
	
###### TUNE KMEANS FEATURE WEIGHTS
# 	best_weights = (-1,-1,-1)
# 	best_score = float('-inf')
# 	one_ten = [1,3,8]
# 	for y_weight in [3]:
# 		for x_weight in [7]:
# 			for z_weight in [9,10,11,12]:
# 				for hue_weight in [3]:
# 					cfg.Y_KMEANS_SCALE = y_weight
# 					cfg.X_KMEANS_SCALE= x_weight
# 					cfg.DEPTH_KMEANS_SCALE = z_weight
# 					cfg.COLOR_KMEANS_SCALE = hue_weight
# 					
# 					#Run segmentation	
# 					extractor = KmeansExtractor(args.input_dir, args.output_dir)
# 					#extractor = AggCluster(args.input_dir, args.output_dir)
# 					predicted_csv = extractor.extractAllObjects() #filepath+name
# 					
# 					
# 					#Compare result with labeled file.
# 					if args.labeled_csv:
# 						print 'Comparing results with labeles.'
# 						best_score, fp, fn = segmentation_error.get_file_accuracy(args.labeled_csv, predicted_csv, cfg.PREDICT_AT_SCALE)
# 						if best_score > best_score:
# 							best_weights = (y_weight, x_weight, z_weight, hue_weight)
# 							best_score = best_score
# 						print '==============='
# 						print 'best_score,fp,fn : ',best_score,fp,fn
# 						print 'y_weight, x_weight, z_weight, hue_weight : ', y_weight, x_weight, z_weight, hue_weight
# 						print '==============='
# 	
# 			
# 	print '=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-'
# 	print 'BEST SCORE : ',best_score
# 	print 'BEST WEIGHTS : ', best_weights
# 		
	
		
	