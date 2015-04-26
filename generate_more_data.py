"""This script looks at a directory with object images, and generates more of them by adding noise, geometric transformations...etc
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
	
	
def add_noise_writer(image, std, prefix):
	"""Adds random noise to given numpy image"""
	image = image.copy()
	noise = np.random.randn(*image.shape)*std
	image = image + noise
	
	output_path = os.path.join(INPUT_DIR, prefix+'_Rnd_'+str(std)+'_COLOR.bmp')
	write(image, output_path)
	
def add_rotation_writer(image, angle, prefix):
	"""Rotates the object around the center, by the specified angle, and writes the result"""
	rows, cols, _ = image.shape
	rotation = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
	image = cv2.warpAffine(image, rotation, (cols,rows)) #returns a copy
	
	output_path = os.path.join(INPUT_DIR, prefix+'_Rot_'+str(angle)+'_COLOR.bmp')

	write(image, output_path)

def add_flip_writer(image, direction, prefix):
	flipcode = 1 if direction=='horizontal' else 0
	image = cv2.flip(image, flipcode)
	output_path = os.path.join(INPUT_DIR, prefix+'_Flip_'+direction[0].upper()+'_COLOR.bmp')
	write(image, output_path)
	
def add_translate_writer(image, offset_x, offset_y, prefix):
	
	rows, cols, _ = image.shape
	translate_matrix = np.float32([[1,0,offset_x],[0,1,offset_y]])
	image = cv2.warpAffine(image, translate_matrix, (cols, rows))
	
	output_path = os.path.join(INPUT_DIR, prefix+'_Trans_'+str(offset_x)+','+str(offset_y)+'_COLOR.bmp')

	write(image, output_path)
	
	

parser = argparse.ArgumentParser(description="""This script looks at a directory with irrelevant images (not containing any relevant objects), and extracts images of different sizes
that should be classified as "non-objects".""")

parser.add_argument('-i',dest='input_dir', default = '/idpdata/frontal_labeledobjects/', help='Directory with objects that we need to generate more images from.')
args = parser.parse_args()

INPUT_DIR = args.input_dir

# FLIP
for colorname, depthname, prefix in idputils.list_images(INPUT_DIR):
	colorfilename = os.path.basename(colorname)
	prefix = '_'.join(colorfilename.split('_')[:-1])
	image = cv2.imread(colorname)
	add_flip_writer(image,'horizontal', prefix)
	
# TRANSLTE original and FLIPPED
for colorname, depthname, prefix in idputils.list_images(INPUT_DIR):
	colorfilename = os.path.basename(colorname)
	prefix = '_'.join(colorfilename.split('_')[:-1])
	image = cv2.imread(colorname)
	
	for offx in [-7,0,7]:
		for offy in [-7,0,7]:
			if offx == 0 and offy == 0: continue
			add_translate_writer(image, offx, offy, prefix)

# ROTATE original and FLIPPED and TRANSLATED
for colorname, depthname, prefix in idputils.list_images(INPUT_DIR):
	colorfilename = os.path.basename(colorname)
	prefix = '_'.join(colorfilename.split('_')[:-1])
	image = cv2.imread(colorname)
	add_rotation_writer(image, 5, prefix)
	add_rotation_writer(image, -5, prefix)
	
	
	
	
	
	
	
	