#!/Users/Mohamed.Ezz/.virtualenvs/idp/bin/python

"""
This is a Standalone Labeling tool suitable for defining a bounding box(es) on objects in an image.
Usage: segmentation_labeler.py /absolute/dir/of/images/to/label/

You point the the tool to a directory containing images, all images in that directory tree will be shown
and the user/labeler should draw multiple rectangles (by dragging) on the desired objects.
While dragging, a rectangle will unfortunately not be drawn on top of the image like in more fancy apps, but the rectangle
will be stored and saved.

When all images are labeled, or ESC is pressed, or when an exception happens, the labeled parts are saved
to segmentation_labels.csv in the same given directory, the CSV has the following header:
image_id, box_id, y1, x1, y2, x2

Images are assumed to be in pairs, color and depth image, name like this:
65430039_COLOR.bmp
65430039_DEPTH.png
##########_COLOR.bmp
##########_DEPTH.png
...etc

When finished labeling, press ESC
"""

import sys, os, csv
import cv,cv2,numpy as np
import shutil
import idputils
from time import time

########### HELPER FUNCTIONS #################
def image_with_rects(image, rects):
	"""Copies given image, draws the given rects on the copy, returns the copy"""
	image_copy = image.copy()
	for rect in rects:
		r1,c1,r2,c2 = rect
		idputils.red_rect(image_copy, r1, c1, r2, c2)
	return image_copy

def refresh_shown_image(image):
	img_rects = image_with_rects(image, current_img_boxes)
	cv2.imshow(WINDOW_NAME,img_rects)
##############################################

WINDOW_NAME = "drawing current_img_boxes"
state = "idle" #or "drawing"
point1 = None #(y,x) first point that defines the bounding box
current_img_boxes = [] #list of tuples (y1,x1,y2,x2)

def on_mouse(event, x, y, flags, image):
	""" Called at every mouse event on the image.
	It implements a small state machine to allow use to Drag/draw a box"""
	global state,point1
	if event == cv.CV_EVENT_LBUTTONDOWN:
		print 'DOWN',x,y
		state = "down"
		point1 = (y,x)
	elif event == cv.CV_EVENT_MOUSEMOVE: 
		state = "drawing"
	elif event == cv.CV_EVENT_LBUTTONUP and state == "drawing":
		state = "idle"
		y1 = point1[0]
		x1 = point1[1]
		x2 = x
		y2 = y
		current_img_boxes.append([y1,x1,y2,x2])
		refresh_shown_image(image)
		print 'UP',current_img_boxes
	else:
		print 'MOUSE DRAW STATE MACHINE CORRUPTED. IN UNKNOWN STATE NOW'

if len(sys.argv) < 2:
	print "Usage: labeler.py /absolute/dir/of/images/to/label/"
	exit(1)

dir = sys.argv[1]
if dir.endswith('/'):
	dir = dir[:-1]

print 'Source dir set to : ',dir

output_filename = os.path.join(dir,'segmentation_labels_'+str(int(time()))+'.csv')
outfile = file(output_filename, 'a')
csv_writer = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_NONE)
csv_writer.writerow(["image_id", "box_id", "y1", "x1", "y2", "x2"])


try:
	cv2.namedWindow(WINDOW_NAME)
	for root, dirnames, filenames in os.walk(dir):
		for filename in filenames:
			if not filename.endswith('_COLOR.bmp'):
				continue
			print filename
			
			prefix = filename.split('_')[0]
			colorname = os.path.join(root,filename)
			colorimage = cv2.imread(colorname)
			while True:
				cv2.setMouseCallback(WINDOW_NAME, on_mouse, colorimage)
				refresh_shown_image(colorimage) #show image with drawn current_img_boxes so far
				keycode = cv2.waitKey(0)
				if keycode == 26: # CTRL+Z (undo)
					current_img_boxes.pop() #remove last drawn box
					print current_img_boxes
				elif keycode == 27: #ESC key
					raise Exception("the catch clause will save images labeled so far and hide this msg")
				else:
					break
			#SAVE BOXES TO FILE
			
			for i,box in enumerate(current_img_boxes):
				box.insert(0,i+1) # +1 to make it 1-indexed
				box.insert(0,prefix)
				#box = img_prefix, box_idx, y1, x1, y2, x2
				csv_writer.writerow(box)
			current_img_boxes = []
			
	outfile.close()
except Exception:
	outfile.close()