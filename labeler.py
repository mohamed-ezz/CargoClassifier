#!/Users/Mohamed.Ezz/.virtualenvs/idp/bin/python

"""
This is a Standalone Labeling tool suitable for 2-class classification labels.
Usage: labeler.py /absolute/dir/of/images/to/label/

You point the the tool to a directory containing images, all images in that directory tree will be shown
and the user/labeler should press LEFT ARROW or RIGHT ARROW. Based on the pressed key, the image (along with 
its corresponding depth image) will be copied to either a folder named LEFT or RIGHT.

The image (and its depth image) will be DELETED from the source directory. To make it easy
to see which images are not yet labeled (the ones that remain in the source directory).

Images are assumed to be in pairs, color and depth image, name like this:
65430039_COLOR.bmp
65430039_DEPTH.png
##########_COLOR.bmp
##########_DEPTH.png
...etc

When finished labeling, press ESC
"""

import sys, os
import cv,cv2,numpy as np
import shutil

if len(sys.argv) < 2:
	print "Usage: labeler.py /absolute/dir/of/images/to/label/"
	exit(1)

dir = sys.argv[1]
if dir.endswith('/'):
	dir = dir[:-1]

print 'Source dir set to : ',dir

leftdir  = os.path.join(os.path.dirname(dir),'LEFT')
if not os.path.exists(leftdir): os.makedirs(leftdir)
rightdir = os.path.join(os.path.dirname(dir),'RIGHT')
if not os.path.exists(rightdir): os.makedirs(rightdir)


for root, dirnames, filenames in os.walk(dir):
	for filename in filenames:
		if not filename.endswith('_COLOR.bmp'):
			continue
		print filename
		
		prefix = filename.split('_')[0]
		colorname = os.path.join(root,filename)
		depthname = os.path.join(root,prefix+'_DEPTH.png')
		
		colorimage = cv2.imread(colorname)
		cv2.imshow('Please press Left or Right arrow', colorimage)
		keycode = cv2.waitKey(0)
		if keycode == 63234: #LEFT ARROW
			print 'Copying',filename,'to',leftdir,'...'
			shutil.copy(colorname, leftdir)
			os.remove(colorname)
			shutil.copy(depthname, leftdir)
			os.remove(depthname)

		elif keycode == 63235: #RIGHT ARROW
			print 'Copying',filename,'to',rightdir,'...'
			shutil.copy(colorname, rightdir)
			os.remove(colorname)
			shutil.copy(depthname, rightdir)
			os.remove(depthname)
		elif keycode == 27: #ESC key
			cv2.destroyAllWindows()
			exit(0)
		else:
			pass #DO NOTHING ABOUT THE SHOWN IMAGE
		
		
		print 'KEYCODE: ',keycode
		cv2.destroyAllWindows()
		
		