#!/Users/Mohamed.Ezz/.virtualenvs/idp/bin/python

helptext="""
This is a Standalone Labeling tool suitable for 2-class classification labels.
Usage: labeler.py /absolute/input_dir/of/images/to/label/

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

If there are no Depth images, it's no problem.
When finished labeling, press ESC
"""

import sys, os
import cv,cv2,numpy as np
import shutil
import idputils
import argparse

parser = argparse.ArgumentParser(description=helptext,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-i",dest="input_dir",required=True,help="The directory containing the images to be labeled. Also the directory that will contain the output folders.")
args = parser.parse_args()


input_dir = args.input_dir
if input_dir.endswith('/'):
	input_dir = input_dir[:-1]

print 'Source input_dir set to : ',input_dir


for root, dirnames, filenames in os.walk(input_dir):
	if root != input_dir: #DONOT traverse the dir tree, only the root.
		continue 
	for filename in filenames:
		if idputils.get_filename_type(filename) != 'COLOR':
			continue
		
		prefix = idputils.get_filename_prefix(filename)
		colorname = os.path.join(root,filename)
		depthname = idputils.to_depth_filename(colorname)
		
		colorimage = cv2.imread(colorname)
		cv2.imshow('Please press a key', colorimage)
		keycode = cv2.waitKey(0)
		
		if keycode == 27: #ESC key
			cv2.destroyAllWindows()
			exit(0)
		else:
			dirname = 'KEYCODE_%i' % keycode
			dirname  = os.path.join(input_dir, dirname) #append full dir path
			if not os.path.exists(dirname):
				os.makedirs(dirname)
			print 'Copying files for',prefix,'to',dirname
			shutil.copy(colorname, dirname)
			os.remove(colorname)
			if os.path.exists(depthname):
				shutil.copy(depthname, dirname)
				os.remove(depthname)

# 		if keycode == 63234: #LEFT ARROW
# 			print 'Copying',filename,'to',leftdir,'...'
# 			shutil.copy(colorname, leftdir)
# 			os.remove(colorname)
# 			if os.path.exists(depthname):
# 				shutil.copy(depthname, leftdir)
# 				os.remove(depthname)
# 
# 		elif keycode == 63235: #RIGHT ARROW
# 			print 'Copying',filename,'to',rightdir,'...'
# 			shutil.copy(colorname, rightdir)
# 			os.remove(colorname)
# 			if os.path.exists(depthname):
# 				shutil.copy(depthname, rightdir)
# 				os.remove(depthname)
# 				
# 		elif keycode == 27: #ESC key
# 			cv2.destroyAllWindows()
# 			exit(0)
# 		else:
# 			pass #DO NOTHING ABOUT THE SHOWN IMAGE
		
		
		print 'KEYCODE: ',keycode
		cv2.destroyAllWindows()
		
		