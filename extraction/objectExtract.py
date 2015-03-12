import os
import cv,cv2
import idputils
import csv
from config import *
import sys
import multiprocessing
import time

class ObjectExtractor:
	
	def __init__(self, imagesdir = None, outputdir=None, experiment_name = '', depth_lo = DEPTH_LO, depth_hi=DEPTH_HI):
		self.imagesdir = imagesdir
		self.outputdir = outputdir
		self.depth_hi = depth_hi
		self.depth_lo = depth_lo
		#just any name to identify the experiment name to make it easier to identify output
		self.experiment_name = experiment_name 
		

		
	def saveObjectImage(self, fname, colorimage, y1,x1,y2,x2):
		""" Given a colorimage (ndarray), extracts/crops a rectangle (given by x1,2 y1,2) and saves it 
		in a separate image file. File written to self.outputdir
		fname : file name including extension. ex: /idpdata/object_1.bmp"""
		objectimage = colorimage[y1:y2+1, x1:x2+1,:]
		fullpath = os.path.join(self.outputdir,fname)
		cv2.imwrite(fullpath,objectimage)
		
	def saveImage(self, fname, colorimage):
		fullpath = os.path.join(self.outputdir,fname)
		cv2.imwrite(fullpath,colorimage)
		
	def extractAllObjects(self):
		"""For each image in self.imagesdir, the segmentation algorithm is run and objects are extracted
		and saved to individual images to self.outputdir"""
		color_depth_prefix = idputils.list_images(self.imagesdir)
		print 'Found total of %i images.' % len(color_depth_prefix)
		if not os.path.exists(self.outputdir):
			os.makedirs(self.outputdir)
		
		#shuffle(color_depth_prefix)
		output_filename = os.path.join(self.outputdir, "segmentation_output.csv")
		csv_file = open(output_filename, 'w')
		csv_writer = csv.writer(csv_file, delimiter=",",quoting=csv.QUOTE_NONE)
		csv_writer.writerow(["image_id", "box_id", "y1", "x1", "y2", "x2"])
		
		t1 = time.time()
		pool = multiprocessing.Pool(processes=cfg.N_PARALLEL_PROCESSES)
		images_rects = pool.map(self, color_depth_prefix) # list of lists. For each item of color_depth_prefix, we get list of rects
		print '\n',time.time()-t1,'seconds'

		for i, rect_list in enumerate(images_rects):
			_, _, prefix = color_depth_prefix[i]
			count = 1
			for y1,x1,y2,x2 in rect_list : #for each found object..write 1 image file
				csv_writer.writerow([prefix,count, y1, x1, y2, x2])
				count +=1
		
		
		
		csv_file.close()
		return output_filename
	
	
	def _threshold_depth(self,colorimage, depthimage):
		"""Masks out the colorimage for pixels with depth out of the desired range defined in
		self.depth_hi and self.depth_low"""
		colorimage[depthimage < self.depth_lo] = 0
		colorimage[depthimage > self.depth_hi] = 0
		return colorimage,depthimage
	
	def _segment(self, colorname, depthname, prefix):
		raise NotImplementedError('_segment function must be overriden by subclasses.')

	def __call__(self, args_tuple):
		"""Workaround to be able to call Pool.map(self._segment, list).
		It solves 2 problems actually: 
		1-instance methods not being picklable, but Instances/Objects themselves are.
		2-not being able to have multi-argument function passed to Pool.map. __call__ acts as a wrapper and does the extraction"""
		colorname, depthname, prefix = args_tuple #Extract
		sys.stdout.write('.')
		sys.stdout.flush()
		return self._segment(colorname, depthname, prefix)
		
		
	
	
	
	
	
	
	
	
	
	