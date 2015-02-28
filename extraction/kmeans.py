from extraction.objectExtract import objectExtractor
import cv2,cv
import numpy as np
import sklearn.preprocessing
from sklearn.cluster import KMeans
import idputils
from operator import add
from scipy import stats

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
STEP_SIZE = 5
	
class kmeans_extractor(objectExtractor):

	
	def _region_grow_score(self, image):
		"""Returns a score for an image region. The score is maximum for regions that are :
		1-Big in area
		2-All have same color (more or less)
		3-Contains minimum number of 0 pixels (in our case, they mean background pixels)
		Image is expected to be 3 channel image after kmeans discretization.
		However all 3 channels have the same value (yes, redundant and needs to be fixed)
		""" 
		#Take 1 channel only of the given image
		image = image[:,:,0]
		
		#Calculate Entropy in bits (for Non-bg pixels only) (the bigger the worse)
		bins_all = np.bincount(np.reshape(image,(image.shape[0]*image.shape[1]))) #at each index i, value=count of value i
		unique = bins_all.nonzero()[0] #indices of nonzero, which = unique values in image 
		counts = bins_all[unique]
		counts = counts[1:]
		entropy = stats.entropy(counts,base=2) #Entropy in bits
		
		
		variance = np.unique(image).shape[0]**7
		bg_pixels_count = (image==0).sum()**1.2
		area = image.size
# 		print 'area,%s' % str(area)
# 		print 'variance,%s' % str(variance)
# 		print 'bg_pixels_count,%s' %str(bg_pixels_count)
# 		print 'entropy,%s'%str(entropy)
		
		return area - entropy*area - bg_pixels_count
		return area - variance - bg_pixels_count
	
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
					y2 = y1+config*70
					x2 = x1+config*30
					score = self._region_grow_score(image[y1:y2+1, x1:x2+1]) 
					if score > max_seed_score:
						max_seed_score = score
						best_rect = np.array([y1,x1,y2,x2])

		##################################
		####### REGION GROWING ###########
		##################################
		def crop(image, rect):
			y1,x1,y2,x2 = rect #tuple
			return image[y1:y2+1, x1:x2+1]
		def grow(rect, full_image, DIRECTION, old_score):
			"""Modifies rect in place by growing it in the specified direction (string)
			according to score on the given image.
			Returns new score if growing improves old_score, or None if growing did not take place because it 
			worsens the score"""
			new_rect = np.array(rect) + STEP_SIZE * GROW_DIRECTION[DIRECTION] #grow/shrink rect
			new_score = self._region_grow_score(crop(full_image,new_rect)) #compute new score
			if new_score > old_score: #assess new score
				rect[0]=new_rect[0]; rect[1]=new_rect[1]; rect[2]=new_rect[2]; rect[3]=new_rect[3] #modify in-place
				return new_score
			else:
				return None
			

# REGION GROWTH ALGO 3 (GROW SHRINK ANY ORDER):	
		score = float('-inf')
		prev_iteration_score = None
		
		while prev_iteration_score != score:
			prev_iteration_score = score
			score = grow(best_rect, image, "GROW_LEFT", score) or score #return 1st non-null value
			score = grow(best_rect, image, "GROW_RIGHT", score) or score
			score = grow(best_rect, image, "GROW_DOWN", score) or score
			score = grow(best_rect, image, "GROW_UP", score) or score
			score = grow(best_rect, image, "SHRINK_LEFT", score) or score #return 1st non-null value
			score = grow(best_rect, image, "SHRINK_RIGHT", score) or score
			score = grow(best_rect, image, "SHRINK_DOWN", score) or score
			score = grow(best_rect, image, "SHRINK_UP", score) or score			
		y1,x1,y2,x2 = best_rect #unpack np array
		return y1,x1,y2,x2
############################################################



##########################

# REGION GROWTH ALGO 2 (GROW ANY ORDER):	
# 		score = float('-inf')
# 		prev_iteration_score = None
# 		
# 		while prev_iteration_score != score:
# 			prev_iteration_score = score
# 			score = grow(best_rect, image, "GROW_LEFT", score) or score #return 1st non-null value
# 			score = grow(best_rect, image, "GROW_RIGHT", score) or score
# 			score = grow(best_rect, image, "GROW_DOWN", score) or score
# 			score = grow(best_rect, image, "GROW_UP", score) or score
# 			
# 		y1,x1,y2,x2 = best_rect #unpack np array
# 		return y1,x1,y2,x2
##########################


# REGION GROWTH ALGO 1 : left,left, left. Right Right. Down,Down,Down. Up Up. End
# 		score = max_seed_score
# 		prev_iteration_score = None
# 		print best_rect
# 		while True:
# 			prev_iteration_score = score
# 			score = grow(best_rect, image, "GROW_LEFT", score) or score #return 1st non-null value
# 			if prev_iteration_score == score: #new score is NOT different
# 				break
# 		print 'LEFT done'
# 		print best_rect
# 		while True:
# 			prev_iteration_score = score
# 			score = grow(best_rect, image, "GROW_RIGHT", score) or score #return 1st non-null value
# 			if prev_iteration_score == score: #new score is NOT different
# 				break
# 		print 'RIGHT done'
# 		print best_rect
# 		while True:
# 			prev_iteration_score = score
# 			score = grow(best_rect, image, "GROW_DOWN", score) or score #return 1st non-null value
# 			if prev_iteration_score == score: #new score is NOT different
# 				break
# 		print 'DOWN done'
# 		print best_rect	
# 		while True:
# 			prev_iteration_score = score
# 			score = grow(best_rect, image, "GROW_UP", score) or score #return 1st non-null value
# 			if prev_iteration_score == score: #new score is NOT different
# 				break			
# 		print 'UP done'
# 		print best_rect
# 		
# 		y1,x1,y2,x2 = best_rect #unpack np array
# 		return y1,x1,y2,x2
##############################################################################
	
	
	def _toKmeansMatrix(self, colorimage,depthimage):
		"""Converts colorimage and depth image, into a Kmeans-ready matrix of shape (n_samples,n_features)
		where n_samples=number of pixels, and n_features=3 color + 3 location=6
		colorimage is RGB image"""
		hsvimage = cv2.cvtColor(colorimage,cv.CV_BGR2HSV)
		height = hsvimage.shape[0]
		width = hsvimage.shape[1]
	
		#Count number of pixels in the desired depth range, to know the 
		#size of matrix to allocate
		#@TODO: SLOW TRICK. 
		pixels = np.ones((height,width),dtype=np.uint8)
		pixels[depthimage < self.depth_lo] = 0
		pixels[depthimage > self.depth_hi] = 0
		n_points = int(pixels.sum()) # number of non-back/foreground pixels. We want to cluster only those
		print 'n_points',n_points
		#Create data matrix
		data = np.zeros((n_points,4))
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

	def _kmeans(self,data):
# 		print 'Start Kmeans'
		model=KMeans(n_clusters=4, n_init=3)
		data = sklearn.preprocessing.scale(data) #returns a copy!
		data[:,2] *= 5 #depth importance
		data[:,3] *= 4 #color importance
		labels = model.fit_predict(data)
# 		print 'Done Kmeans'
		return labels
	
	def _segment(self, colorimage, depthimage, prefix):
		
		colorimage,depthimage = self._threshold_depth(colorimage, depthimage)
		#data is n_pixels x n_features matrix. Contains only foreground pixels.
		#Each row in data matrix is (y, x, z, hue)
# 		print 'Preparing data matrix'
		data = self._toKmeansMatrix(colorimage, depthimage) 
		labels = self._kmeans(data)
		#Put segmentation result onto image
		segmentimage = colorimage.copy() #image that will carry cluster numbers
		for i in range(data.shape[0]):
			graylevel = 84*(labels[i]+1)
			#@TODO: make segmentimage single channel !!
			segmentimage[data[i][0], data[i][1]] = (graylevel,graylevel,graylevel)
		
		segmentimage = cv2.medianBlur(segmentimage, 61)
		
		segmentimage_orig = segmentimage.copy()
		rects = []
		
		ay1,ax1,ay2,ax2 = self._find_least_variance_box(segmentimage)
		segmentimage[ay1:ay2+1, ax1:ax2+1] = 0
		idputils.red_rect(colorimage, ay1,ax1,ay2,ax2)
		idputils.red_rect(segmentimage_orig, ay1,ax1,ay2,ax2)
		rects.append((ay1,ax1,ay2,ax2))
		
		by1,bx1,by2,bx2 = self._find_least_variance_box(segmentimage)
		segmentimage[by1:by2+1, bx1:bx2+1] = 0
		idputils.red_rect(colorimage, by1,bx1,by2,bx2)
		idputils.red_rect(segmentimage_orig, by1,bx1,by2,bx2)
		rects.append((by1,bx1,by2,bx2))
		
# 		cy1,cx1,cy2,cx2 = self._find_least_variance_box(segmentimage)
# 		segmentimage[cy1:cy2+1, cx1:cx2+1] = 0
# 		idputils.red_rect(colorimage,cy1,cx1,cy2,cx2)
# 		rects.append((cy1,cx1,cy2,cx2))
		
		#self.saveImage(prefix+'_COLOR.bmp', colorimage)
		#self.saveImage(prefix+'_SEGMENTS.png', segmentimage_orig)
		#idputils.imshow(colorimage,'color segment result')
		#idputils.imshow(segmentimage_orig,'segment result')
		
		
		return rects
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		