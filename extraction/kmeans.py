from extraction.objectExtract import objectExtractor
import cv2,cv
import numpy as np
import sklearn.preprocessing
from sklearn.cluster import KMeans
import idputils

class kmeans_extractor(objectExtractor):
	
	def _region_grow_score(self, image):
		"""Returns a score for an image region. The score is maximum for regions that are :
		1-Big in area
		2-All have same color (more or less)
		3-Contains minimum number of 0 pixels (in our case, they mean background pixels)
		"""
		variance = np.unique(image).shape[0]**5 + image.shape[0]
		bg_pixels_count = (image==0).sum()**1.2
		area = image.size
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
		max_score = -9999999
		best_rect = () #tuple (y1,x1,y2,x2)
		for y1 in xrange(0,height-hsize,step):
			for x1 in xrange(0,width-wsize,step):
				for config in xrange(1,2):
					y2 = y1+config*70
					x2 = x1+config*30
					score = self._region_grow_score(image[y1:y2+1, x1:x2+1]) 
					if score > max_score:
						max_score = score
						best_rect = y1,x1,y2,x2

		##################################
		####### REGION GROWING ###########
		##################################
		prev_score = -9999999
		STEP = 5
		while True:
			y1,x1,y2,x2 = best_rect
			x1 -= STEP
			new_score = self._region_grow_score(image[y1:y2+1, x1:x2+1])
			if prev_score > new_score or x1 < 10: break
			else:
				best_rect = y1,x1,y2,x2
				prev_score = new_score
	
		while True:
			y1,x1,y2,x2 = best_rect
			x2 += STEP
			new_score = self._region_grow_score(image[y1:y2+1, x1:x2+1])
			if prev_score > new_score or x2 > image.shape[1]-10: break
			else:
				best_rect = y1,x1,y2,x2
				prev_score = new_score
	
		while True:
			y1,x1,y2,x2 = best_rect
			y2 += STEP
			new_score = self._region_grow_score(image[y1:y2+1, x1:x2+1])
			if prev_score > new_score or y2 > image.shape[0]-10: break
			else:
				best_rect = y1,x1,y2,x2
				prev_score = new_score
	
		while True:
			y1,x1,y2,x2 = best_rect
			y1 -= STEP
			new_score = self._region_grow_score(image[y1:y2+1, x1:x2+1])
			if prev_score > new_score or y1 < 10: break
			else:
				best_rect = y1,x1,y2,x2
				prev_score = new_score
		
		return y1,x1,y2,x2
	
	
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
		print 'Start Kmeans'
		model=KMeans(n_clusters=4, n_init=3)
		data = sklearn.preprocessing.scale(data) #returns a copy!
		data[:,2] *= 5 #depth importance
		data[:,3] *= 4 #color importance
		labels = model.fit_predict(data)
		print 'Done Kmeans'
		return labels
	
	def _segment(self, colorimage, depthimage, prefix):
		
		colorimage,depthimage = self._threshold_depth(colorimage, depthimage)
		#data is n_pixels x n_features matrix. Contains only foreground pixels.
		#Each row in data matrix is (y, x, z, hue)
		print 'Preparing data matrix'
		data = self._toKmeansMatrix(colorimage, depthimage) 
		labels = self._kmeans(data)
		#Put segmentation result onto image
		segmentimage = colorimage.copy() #image that will carry cluster numbers
		for i in range(data.shape[0]):
			graylevel = 84*(labels[i]+1)
			segmentimage[data[i][0], data[i][1]] = (graylevel,graylevel,graylevel)
		
		segmentimage = cv2.medianBlur(segmentimage, 31)
		
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
		
		self.saveImage(prefix+'_COLOR.bmp', colorimage)
		self.saveImage(prefix+'_SEGMENTS.png', segmentimage_orig)
		#idputils.imshow(colorimage,'color segment result')
		#idputils.imshow(segmentimage_orig,'segment result')
		
		
		return rects
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		