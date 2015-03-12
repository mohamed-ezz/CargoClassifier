"""
Small standalone script to show bounding boxes specified in a csv segmentation file for a given image.

Script takes an image path (or wildcard matching many images), and a path to segmentation csv.

This tool will view the image, and draw the segmentation boxes on top of it.

Examples:
python show_segmentation.py -i /idpdata/frontal/1001923000_COLOR.bmp -s /idpdata/frontal/segmentation_labels.csv
python show_segmentation.py -i /idpdata/frontal/*.bmp -s /idpdata/frontal/segmentation_labels.csv 
"""

import argparse
import idputils
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i',dest='imagepath', nargs='+', help="Path to a single image, or a wildcard path matching multiple images.")
parser.add_argument('-s',dest='csvpath')
args = parser.parse_args()

prefix_box = idputils.read_segmentation_csv(args.csvpath)
for colorimagefilePATH in args.imagepath:
	color_image = cv2.imread(colorimagefilePATH) #numpy ndarray
	our_prefix = idputils.get_filename_prefix(colorimagefilePATH)
	boxes = prefix_box[our_prefix]
	
	for box in boxes:
		idputils.red_rect(color_image, *box)
	print colorimagefilePATH
	idputils.imshow(color_image, "Path: "+colorimagefilePATH)