import os
import cv,cv2
import idputils
import csv

class objectExtractor:
	
	DEPTH_HI = 3500
	DEPTH_LO = 1750
	
	
	def __init__(self, imagesdir = None, outputdir=None, experiment_name = '', depth_lo = DEPTH_LO, depth_hi=DEPTH_HI):
		self.imagesdir = imagesdir
		self.outputdir = outputdir
		self.depth_hi = depth_hi
		self.depth_lo = depth_lo
		#just any name to identify the experiment name to make it easier to identify output
		self.experiment_name = experiment_name 
		
	def listImage(self):
		""" Returns a list of tuples, one for each image prefix, found in the self.imagesdir dir tree (recursively)
		A tuple looks like this : (colorimage filepath, depthimage filepath, prefix)
		"""
		result = [] #list of pairs of file names (color and depth). Absolute paths
				
		for root,dirnames,filenames in os.walk(self.imagesdir):
			colorfnames = filter(lambda fname: fname.endswith('COLOR.bmp') , filenames)
			prefixes = map(lambda fname: fname.split('_')[0], colorfnames)
			abspath_pairs = map(lambda prefix: (os.path.join(root,prefix)+'_COLOR.bmp'  , os.path.join(root,prefix)+'_DEPTH.png', prefix)
								, prefixes)
			result.extend(abspath_pairs)
		return result
		
	def saveObjectImage(self, fname, colorimage, y1,x1,y2,x2):
		""" Given a colorimage (ndarray), extracts/crops a rectangle (given by x1,2 y1,2) and saves it 
		in a separate image file. File written to self.outputdir
		fname : file name including extension. ex: /idpdata/object_1.bmp"""
		object = colorimage[y1:y2+1, x1:x2+1,:]
		fullpath = os.path.join(self.outputdir,fname)
		cv2.imwrite(fullpath,object)
		
	def saveImage(self, fname, colorimage):
		fullpath = os.path.join(self.outputdir,fname)
		cv2.imwrite(fullpath,colorimage)
		
	def extractAllObjects(self):
		"""For each image in self.imagesdir, the segmentation algorithm is run and objects are extracted
		and saved to individual images to self.outputdir"""
		color_depth_prefix = self.listImage()
		print 'Found total of %i images.' % len(color_depth_prefix)
		if not os.path.exists(self.outputdir):
			os.makedirs(self.outputdir)
		
		#shuffle(color_depth_prefix)
		done_count = 0
		csv_file = open(os.path.join(self.outputdir,"segmentation_output_"+self.experiment_name+".csv"), 'w')
		csv_writer = csv.writer(csv_file, delimiter=",",quoting=csv.QUOTE_NONE)
		csv_writer.writerow(["image_id", "box_id", "y1", "x1", "y2", "x2"])
		
		for colorname,depthname,prefix in color_depth_prefix:
			print colorname
			colorimage = cv2.imread(colorname)
			depthimage = cv2.imread(depthname,-1) #-1 to read data as is.
			#print 'Processing :',colorname,depthname,prefix
			rects = self._segment(colorimage, depthimage, prefix)
			done_count+=1
			print '%i/%i' %(done_count, len(color_depth_prefix)),prefix
			
			
			count = 1
			for y1,x1,y2,x2 in rects : #for each found object..write 1 image file
				csv_writer.writerow([prefix,count, y1, x1, y2, x2])
				#self.saveObjectImage(os.path.join(self.outputdir,prefix+'_'+str(count)+'.png'), colorimage, y1,x1,y2,x2)
				count +=1
		
		csv_file.close()
	
	def _threshold_depth(self,colorimage, depthimage):
		"""Masks out the colorimage for pixels with depth out of the desired range defined in
		self.depth_hi and self.depth_low"""
		colorimage[depthimage < self.depth_lo] = 0
		colorimage[depthimage > self.depth_hi] = 0
		return colorimage,depthimage
	
	def _segment(self, colorimage, depthimage):
		raise NotImplementedError('_segment function must be overriden by subclasses.')
	

	
	
	
	
	
	
	
	
	
	
	
	