"""
Small standalone script to show bounding boxes specified in a csv segmentation file for a given image.
It also prints stats about the region scores, and number of objects per image.

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
import numpy as np
from extraction.kmeans import KmeansExtractor
from extraction import config as cfg

parser = argparse.ArgumentParser()
parser.add_argument('-i',dest='imagepath', nargs='+', help="Path to a single image, or a wildcard path matching multiple images.")
parser.add_argument('-s',dest='csvpath')
parser.add_argument('--stats-only',dest='statsonly', default=False, action = "store_true", help="Only print stats, do not show images.")
parser.add_argument('--scale', dest='scale',type=float, default=cfg.PREDICT_AT_SCALE, help='How much coordinates in the CSV are scaled down. If value = 0.4, then coordinates will be scaled up by 2.5 to show on the original image.')
parser.add_argument('--max-score', dest='maxscore',type=float, default=float('inf'), help='Script will only show boxes that have score less than this value.')
parser.add_argument('--min-score', dest='minscore',type=float, default=float('-inf'), help='Script will only show boxes that have score more than this value.')
args = parser.parse_args()

prefix_box = idputils.read_segmentation_csv(args.csvpath)

array_n_boxes = []
array_scores = []
km = KmeansExtractor()


for colorimagefilePATH in args.imagepath:
	depthimagefilePATH = idputils.to_depth_filename(colorimagefilePATH)
	our_prefix = idputils.get_filename_prefix(colorimagefilePATH)
	boxes = prefix_box[our_prefix]
	clustered_image = km._segment(colorimagefilePATH, depthimagefilePATH, our_prefix, output_clustered_image=True)
	if not args.statsonly:
		color_image = cv2.imread(colorimagefilePATH) #numpy ndarray
		show_image = False
		for box in boxes:
			y1,x1,y2,x2 = np.array(box) * cfg.PREDICT_AT_SCALE
			score = km._region_grow_score(clustered_image[y1:y2+1, x1:x2+1])

			if score < args.maxscore and score > args.minscore :
				show_image = True
				idputils.red_rect(color_image, *box)
		print colorimagefilePATH
		if show_image:
			idputils.imshow(clustered_image, "clustered")
			idputils.imshow(color_image, "Path: "+colorimagefilePATH)
	
	array_n_boxes.append(len(boxes))
	for box in boxes:
		y1,x1,y2,x2 = np.array(box) * cfg.PREDICT_AT_SCALE
		score = km._region_grow_score(clustered_image[y1:y2+1, x1:x2+1])
		array_scores.append(score)
	
	
array_n_boxes = np.array(array_n_boxes)
array_scores  = np.array(array_scores)

print """
Stats:
Number of images : %i
Number of objects: %i

Number of objects is always one of : %s
Mean number of objects : %f
Std number of objects  : %f
Var number of objects  : %f

Scores max   : %f
Scores min   : %f
Scores mean  : %f
Scores Std   : %f
Scores Var   : %f
Scores Median: %f

"""% (
len(args.imagepath),
len(array_scores),
str(np.unique(array_n_boxes)),
array_n_boxes.mean(),
array_n_boxes.std(),
array_n_boxes.var(),
array_scores.max(),
array_scores.min(),
array_scores.mean(),
array_scores.std(),
array_scores.var(),
np.median(array_scores)
)

	







	