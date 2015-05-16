"""
Example:
python crop_segmented_objects -i /idpdata/frontal/*_COLOR.bmp -s /idpdata/frontal/segmentation_label.csv -o /idpdata/frontal_objects/
Crops objects from images and saves each object to an image file.

Takes a list of color_image names and a segmentation csv file.
Then each object (defined in the csv file for each image) is written to an output directory as a separate image file.
If a depth image was found beside the color image, a depth version of the object will be written to a separate file too.


Suppose that image
1234_COLOR.bmp
1234_DEPTH.png

has 3 objects, then the output will be:
1234_COLOR_1.bmp
1234_COLOR_2.bmp
1234_COLOR_3.bmp

1234_DEPTH_1.png
1234_DEPTH_2.png
1234_DEPTH_3.png

"""

import argparse
import idputils
import cv2
import os
import numpy as np

	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i',dest='imagepath', nargs='+', help="Path to a single color image, or a wildcard path matching multiple images.")
	parser.add_argument('-s',dest='csvpath')
	parser.add_argument('-o',dest='output_dir')
	parser.add_argument('-scale',dest='scale', type=float, default = 1, help="The scale at which coordinates are saved in the csv. E.g: if -scale 0.4, then coordinates will be scaled up by 1/0.4=2.5")
	args = parser.parse_args()
	
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	prefix_box = idputils.read_segmentation_csv(args.csvpath)
	for colorimagefilePATH in args.imagepath:
		our_prefix = idputils.get_filename_prefix(colorimagefilePATH)
		
		#Get depthfile full path
		depthimagefilePATH = idputils.to_depth_filename(colorimagefilePATH)
		
		boxes = prefix_box[our_prefix]
		colorimagefilename = os.path.basename(colorimagefilePATH)
		depthimagefilename = os.path.basename(depthimagefilePATH)
		
		count = 1
		print 'Cropping %i from image with id %s' % (len(boxes), str(our_prefix))
		for box in boxes:
			box = np.array(box)*(1/args.scale)
			
			idputils.crop_to_file(colorimagefilePATH, 
					os.path.join(args.output_dir, idputils.to_object_filename(colorimagefilename, count)), *box)
			idputils.crop_to_file(depthimagefilePATH, 
					os.path.join(args.output_dir, idputils.to_object_filename(depthimagefilename, count)), *box)
			count += 1
			
			
			
			
			
			
			
			