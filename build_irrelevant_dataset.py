helptxt="""This script looks at a directory with irrelevant images (not containing any relevant objects), and extracts images of different sizes
that should be classified as "non-objects"."""

import argparse
import cv2,cv
import idputils
import os,sys

parser = argparse.ArgumentParser(description=helptxt)

parser.add_argument('-i',dest='input_dir', default = '/idpdata/All data/irrelevant', help='Directory containing images with no objects, that will be used to extract subimages of different sizes.')
parser.add_argument('-o', dest='output_dir', default = '/idpdata/frontal_labeledobjects/IRRELEVANT/', help='Directory to write output images.')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)

#STEP = 50
#SHAPES = [(256,75), (100,100),
#		  (128,90), (75,192)]

STEP = 100
SHAPES = [(140,75), (75,140), (100,100)]

for colorname, depthname, prefix in idputils.list_images(args.input_dir):
	colorimage = cv2.imread(colorname)
	depthimage = cv2.imread(depthname, -1)
	height, width = colorimage.shape[0], colorimage.shape[1]
	
	print ''
	print colorname
	count=1
	#we sample from the bottom half of the images bcoz in our pictures, all background (top of img) is thresholded by depth already,
	#and we usually will encounter non-objects from the part that passes the depth threshold (which is mostly the ground at the bottom of the image) 
	for y in range(height/2, height, STEP): 
		for x in range(0, width, STEP):
			for obj_height, obj_width in SHAPES:
				if y+obj_height > height - 1 or x+obj_width > width - 1: continue
				count +=1
				
				box = y, x, y+obj_height, x+obj_width
				sys.stdout.write('.');sys.stdout.flush()
				idputils.crop_to_file_fromimage(colorimage, 
						os.path.join(args.output_dir, prefix+'_'+str(count)+'_COLOR.bmp'), *box)
				idputils.crop_to_file_fromimage(depthimage, 
						os.path.join(args.output_dir, prefix+'_'+str(count)+'_DEPTH.png'), *box)
				
	
