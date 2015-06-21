helptxt="""This script looks at a directory with object images, and generates more of them by adding noise, geometric transformations...etc
The newly generated images goes into the same input folder, grouped in a directory named 'synthetic' """

import argparse
import cv2,cv
import idputils
import os,sys
import numpy as np

def write(image, output_path):
	
	print output_path
	output_dir = os.path.dirname(output_path)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	cv2.imwrite(output_path, image)
	sys.stdout.write('.')
	
	
def add_noise_writer(image, std, prefix, DIR):
	"""Adds random noise to given numpy image"""
	image = image.copy()
	noise = np.random.randn(*image.shape)*std
	image = image + noise
	
	output_path = os.path.join(DIR, prefix+'_Rnd_'+str(std)+'_COLOR.bmp')
	write(image, output_path)
	
def add_rotation_writer(image, angle, prefix, DIR):
	"""Rotates the object around the center, by the specified angle, and writes the result"""
	rows, cols, _ = image.shape
	rotation = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
	image = cv2.warpAffine(image, rotation, (cols,rows)) #returns a copy
	
	output_path = os.path.join(DIR, prefix+'_Rot_'+str(angle)+'_COLOR.bmp')

	write(image, output_path)

def add_flip_writer(image, direction, prefix, DIR):
	flipcode = 1 if direction=='horizontal' else 0
	image = cv2.flip(image, flipcode)
	output_path = os.path.join(DIR, prefix+'_Flip_'+direction[0].upper()+'_COLOR.bmp')
	write(image, output_path)
	
def add_translate_writer(image, offset_x, offset_y, prefix, DIR):
	
	rows, cols, _ = image.shape
	translate_matrix = np.float32([[1,0,offset_x],[0,1,offset_y]])
	image = cv2.warpAffine(image, translate_matrix, (cols, rows))
	
	output_path = os.path.join(DIR, prefix+'_Trans_'+str(offset_x)+','+str(offset_y)+'_COLOR.bmp')

	write(image, output_path)
	
	

def affine_main(args):
	INPUT_DIR = args.input_dir
	
	# FLIP
	for colorname, depthname, prefix in idputils.list_images(INPUT_DIR):
		colorfilename = os.path.basename(colorname)
		prefix = '_'.join(colorfilename.split('_')[:-1])
		image = cv2.imread(colorname)
		add_flip_writer(image,'horizontal', prefix, INPUT_DIR)
	
	return	
	# TRANSLTE original and FLIPPED
	for colorname, depthname, prefix in idputils.list_images(INPUT_DIR):
		colorfilename = os.path.basename(colorname)
		prefix = '_'.join(colorfilename.split('_')[:-1])
		image = cv2.imread(colorname)
		
		for offx in [-7,0,7]:
			for offy in [-7,0,7]:
				if offx == 0 and offy == 0: continue
				add_translate_writer(image, offx, offy, prefix, INPUT_DIR)
	
	
	#GENERATE PERTURBED CROPS from labeled data
	
	
	
	# ROTATE original and FLIPPED and TRANSLATED
	# for colorname, depthname, prefix in idputils.list_images(INPUT_DIR):
	# 	colorfilename = os.path.basename(colorname)
	# 	prefix = '_'.join(colorfilename.split('_')[:-1])
	# 	image = cv2.imread(colorname)
	# 	add_rotation_writer(image, 5, prefix)
	# 	add_rotation_writer(image, -5, prefix)
	

def perturb(y1,x1,y2,x2):
	""" Perturb the size of the given rectangle
	"""
	#How many sides to perturb?
	n_sides = np.random.random_integers(1,3)

	for _ in range(n_sides):
		#Choose which side to perturb now
		side = np.random.choice(['top','left','bottom','right'],1)[0]
		#Choose Shrink or Extend
		sign = np.random.choice([-1,1,1,-1],1)[0] #returns 1 or -1.
		#Proportion to be added/subtracted (extend/shrink)
		proportion = abs(np.random.normal(0.15,0.07)) #abs will bias distribution by mirroring negative values to be positive
		#Apply perturb
		if side == 'top':
			length = abs(y2-y1)
			y1 += sign*proportion*length
		elif side == 'left':
			length = abs(x2-x1)
			x1 += sign*proportion*length
		elif side == 'bottom':
			length = abs(y2-y1)
			y2 += sign*proportion*length
		elif side == 'right':
			length = abs(x2-x1)
			x2 += sign*proportion*length

			
	def bound(c, lo, hi):
		return int(min(hi,max(c,lo)))
	
	return bound(y1,0,480), bound(x1,0,640), bound(y2,0,480), bound(x2,0,640)	
	
		

def perturb_main(args):
	PERTURBS_PER_OBJECT = 16
	
	for d in ['BARREL','BLUEBOX','BROWNBOX','irrelevant']:
		dir_to_create = os.path.join(args.input_dir,d) 
		if not os.path.exists(dir_to_create):
			os.mkdir(dir_to_create)

	prefix_boxAndType = idputils.read_segmentation_withobjecttype_csv(args.seglabel_path) # {prefix: [(y1,x1,y2,x2,label_as_str)]
	for prefix in prefix_boxAndType.keys():
		objid = 0
		for y1,x1,y2,x2,labelstr in prefix_boxAndType[prefix]:
			objid +=1
			for i in range(PERTURBS_PER_OBJECT):
				ny1,nx1,ny2,nx2 = perturb(y1,x1,y2,x2)
				output_filename = os.path.join(args.input_dir, labelstr, prefix+'_'+str(objid)+'_P'+str(i+1)+'_COLOR.bmp')
				print 'Writing to:',output_filename, ny1,nx1,ny2,nx2
				idputils.crop_to_file(os.path.join(args.input_dir,prefix+'_COLOR.bmp'), output_filename, ny1, nx1, ny2, nx2)
				
				
				
parser = argparse.ArgumentParser(description=helptxt)
subparsers = parser.add_subparsers()
#SUBCOMMAND 1
affine_parser = subparsers.add_parser('affine', help='Apply affine transformations to existing cropped object images.')
affine_parser.add_argument('-i',dest='input_dir', default = '/idpdata/frontal_labeledobjects/', help='Directory with objects that we need to generate more images from.')
affine_parser.set_defaults(func=affine_main)
#SUBCOMMAND 2
perturb_parser = subparsers.add_parser('perturb', help="""Crops perfectly segmented/labeled objects, by randomly extending the perfect bounding box, 
to mimic segmentation algorithm behaviour on unseen data. Purpose is to generate data used for training, that looks as much as possible 
as the input to classifiers coming out of the segmentation algorithm. The generated data will be written to a folder per object type under the input folder directory.""")
perturb_parser.add_argument('-l',dest='seglabel_path', default = '/idpdata/frontal/segmentation_labels_withobjects.csv', help='CSV path of segmentation+objectType label.')
perturb_parser.add_argument('-i',dest='input_dir', default = '/idpdata/frontal/', help='Directory with full that will be used to cropped perturbed bounding boxes.')
perturb_parser.set_defaults(func=perturb_main)

args = parser.parse_args()
args.func(args)
	

	
	
	
	
	
	
	
	