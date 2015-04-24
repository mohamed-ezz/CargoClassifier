import cv,cv2
from collections import defaultdict
import os
import glob
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np


def imshow(img,title=''):
	cv2.imshow(title,img)
	disp()

def disp():
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def red_rect(img,r1,c1,r2,c2,width=2):
	""" Draws a red rectangle on given 3 channel image """
	w=width
	#vertical lines
	img[r1:r2,c1:c1+w] = (0,0,255)
	img[r1:r2,c2:c2+w] = (0,0,255)
	#horizontal lines
	img[r1:r1+w,c1:c2] = (0,0,255)
	img[r2:r2+w,c1:c2] = (0,0,255)

def read_segmentation_csv(filename):
	"""Given a segmentation csv file, returns a dictionary {image_id: list of boxes}"""
	f = open(filename, 'r')
	output = defaultdict(list)
	for line in f:
		if line.startswith('image_id,'): continue #skip header
		prefix, box_id, y1, x1, y2, x2 = map(int,line.split(',')) #parse
		output[str(prefix)].append((y1,x1,y2,x2))
	
	return output
	

def list_images(imagesdir):
	""" Returns a list of tuples, one for each image prefix, found in the self.imagesdir dir tree (recursively)
	A tuple looks like this : (colorimage filepath, depthimage filepath, prefix)
	"""
	if imagesdir[-1] != '/':
		imagesdir += '/'
	
	colorfnames = glob.glob(imagesdir+'*COLOR*')
	depthfnames = map(to_depth_filename, colorfnames)
	prefixes    = map(get_filename_prefix, colorfnames)
	return zip(colorfnames, depthfnames, prefixes)
	
	

def crop_to_file_fromimage(image, output_filename, y1,x1,y2,x2):
	cropped_image = image[y1:y2+1, x1:x2+1]
	cv2.imwrite(output_filename, cropped_image)
	
def crop_to_file(fullimage_filename, output_filename, y1,x1,y2,x2):
	if get_filename_type(fullimage_filename) == 'COLOR':
		image = cv2.imread(fullimage_filename) #numpy ndarray
	elif get_filename_type(fullimage_filename) == 'DEPTH':
		image = cv2.imread(fullimage_filename, -1) #numpy ndarray
	else:
		raise ValueError('given filename does not seem to be COLOR or DEPTH image: %s' % fullimage_filename)
	cropped_image = image[y1:y2+1, x1:x2+1]
	cv2.imwrite(output_filename, cropped_image)

######### IMAGE FILE NAMING HELPER FUNCTIONS ############

def is_object_filename(filename):
	"""Returns true if the filename given looks like a cropped image of an Object (box)
	This is detected from the filename if it's like: 2850923_1_COLOR.bmp or 2850923_1_DEPTH.png 
	"""
	filename = os.path.basename(filename)
	return len(filename.split("_")) == 3

def get_filename_type(filename):
	"""Returns 'COLOR' or 'DEPTH' """
	filename = os.path.basename(filename)
	return filename.split("_")[-1].split(".")[0] # split on . to remove extension
	
def get_filename_prefix(filename):
	"""Extracts prefix/image_id given a filename of a full image (e.g 1234_COLOR.bmp) or 
	an object file (e.g 1234_1_COLOR.bmp)"""
	filename = os.path.basename(filename)
	return filename.split("_")[0]


def get_filename_extension(filename):
	""" Returns .bmp or .png 
	note that it includes the dot"""
	filename = os.path.basename(filename)
	return filename[-4:]

def to_depth_filename(filename):
	
	directory = os.path.dirname(filename)
	filename = os.path.basename(filename)
	if get_filename_type(filename) == 'COLOR':
		return os.path.join(directory, filename.replace('COLOR.bmp','DEPTH.png') )
	else:
		raise ValueError("Given filename is not a COLOR image filename: %s" % filename)
	
def to_color_filename(filename):
	
	directory = os.path.dirname(filename)
	filename = os.path.basename(filename)
	if get_filename_type(filename) == 'DEPTH':
		return os.path.join(directory, filename.replace('DEPTH.png','COLOR.bmp') )
	else:
		raise ValueError("Given filename is not a DEPTH image filename: %s" % filename)
	
def to_object_filename(filename, objectindex):
	"""From a color image name (e.g 123_COLOR.bmp), builds the filename for an extracted(for cropping) object
	with index = objectindex"""
	origargument = filename
	directory = os.path.dirname(filename)
	filename = os.path.basename(filename)
	if is_object_filename(filename):
		raise ValueError("The given filename is already an object filename: %s" % origargument)
	
	prefix = get_filename_prefix(filename)
	objectindex = str(objectindex)
	imgtype = get_filename_type(filename)
	ext = get_filename_extension(filename)
	return os.path.join(directory, '%s_%s_%s%s' % (prefix, objectindex, imgtype, ext))
	
	
def rects_to_objectimages(colorimage, rects):
	"""Crops a given image to objects, given the rects(box positions to crop at)
	rects = [(y1,x1,y2,x2), (y1,x1,y2,x2), ...etc]. Returns a list of images (numpy arrays)"""
	images = []
	for rect in rects:
		y1,x1,y2,x2 = rect
		images.append(colorimage[y1:y2+1, x1:x2+1])
	return images
		
		
###### PLOTTING ######
def scatter3d(x,y,z=None,labels=None,colors=None,sample_percentage=None,outputfile=None,show=False):
	""" sample_percentage : if not None, only a sample of the full given data will be plotted. percentage here is 0 to 1 indicates how
	much to sample. Sampling works for 3d plotting only (if z is not None) """
	
	if z == None:
		plt.scatter(x,y,color=colors) #No need to sampple in 2D plots (you don't need to rotate and play around...it's just an image)
	else:
		print 'Length before sampling:',len(x),' ',len(y), ' ', len(z)
		if  sample_percentage:
			count = len(x)
			zipped = zip(x,y,z,colors) if colors else zip(x,y,z) #build data points instead of lists of x, list of y, list of z
			random.shuffle(zipped)
			zipped = zipped[:int(count*sample_percentage)] #shuffle and take first tenth of the data
			if colors:
				x,y,z,colors = zip(*zipped)
			else:
				x,y,z = zip(*zipped)
		print 'After sampling:', len(x),' ',len(y), ' ', len(z)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		if labels != None:
			ax.set_xlabel(labels[0])
			ax.set_ylabel(labels[1])
			ax.set_zlabel(labels[2])
		ax.scatter(x,y,z,color=colors)

	plt.axis('equal')
	if outputfile == None or show:
		plt.show()
	else:
		plt.savefig(outputfile)
	plt.close()
	
	
	
	
def plot_prec_recall(y_list, y_pred_prob_list, labels_list, title, filename=None):
	"""Takes a list of one or more y (ground truth labels), and predicted probabilities y_pred_prob (numpy arrays)
	and produces a precision recall plot. Also a label for each line is given in labels_list.
	The plot will have the given title and saved to filename"""
	
	zipped = zip(y_list, y_pred_prob_list, labels_list)
	for y, y_pred_prob, label in zipped:
		precision, recall, _ = precision_recall_curve(y, y_pred_prob)
		print 'data points on Prec-Rec : %i' % len(recall)
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim(0,1)
		plt.xlim(0,1)
		plt.plot(recall,precision, label=label)
		
	plt.suptitle(title)
	plt.legend(loc=3)
	if filename:
		plt.savefig(filename)
	else:
		plt.show()
	plt.clf() # clear plot
	
	
	
	
	
	