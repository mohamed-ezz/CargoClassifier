"""
This Standalone script calculates the get_box_accuracy of certain segmentation algorithm compared to ground truth(optimal) segmentation
The input is 2 segmentation csv files. Each csv has the same format as the following:
image_id/prefix , box_id, y1, x1, y2, x2

The get_box_accuracy is a number ranging from 0 to 1.0.
The number detected boxes may be different than the true number of objects (as per the labeling)
there could be more detected boxes than there is, or less.

The get_box_accuracy out of two files is calculated as follows :
For each image in the predicted: get get_box_accuracy of each predicted bounding box as the ratio of overlap area
over the total are of the 2 boxes (predicted and true). Best best_score is 1, worst is 0.
Then each extra box predicted, penalizes the best_score by -1 
And each missing box (not predicted), penalizes the best_score by -1.

Example:
python segmentation_error.py -p /idpdata/seg_output/kmeans_growth/segmentation_output.csv -l /idpdata/frontal/segmentation_labels.csv --show-false-negatives -i /idpdata/frontal/ -s 0.4
"""
import sys
from collections import defaultdict
import idputils
from argparse import ArgumentParser
import cv2,cv
	
def get_area(box):
	y1, x1, y2, x2 = box
	return abs((y2-y1)*(x2-x1))


def get_overlap(boxA,boxB, scale_B=1):
	"""Get number of pixels of 2 rectangles, each defined by 2 points"""
	#Ensure that first point is top left of the second point for Box A
	ay1, ax1, ay2, ax2 = min(boxA[0],boxA[2]), min(boxA[1],boxA[3]), max(boxA[0],boxA[2]), max(boxA[1],boxA[3])
	#same for Box B
	by1, bx1, by2, bx2 = min(boxB[0],boxB[2]), min(boxB[1],boxB[3]), max(boxB[0],boxB[2]), max(boxB[1],boxB[3])
	by1, bx1, by2, bx2 = by1/scale_B, bx1/scale_B, by2/scale_B, bx2/scale_B
	
	x_overlap = max(0, min(ax2,bx2) - max(ax1,bx1))
	y_overlap = max(0, min(ay2,by2) - max(ay1,by1))
	overlap = x_overlap * y_overlap
	return overlap
	
	
	
def get_box_accuracy(true_box, predicted_box, scale_predicted):
	tarea = get_area(true_box)
	parea = get_area(predicted_box) / scale_predicted / scale_predicted
	
	overlaparea = get_overlap(true_box, predicted_box, scale_predicted)
	#print tarea,parea,overlaparea
	best_score = (overlaparea*2.0) / (tarea+parea)
	return best_score

def get_file_accuracy(filename_true, filename_predicted, scale_predicted, show_false_negatives_from_dir=None):
	"""Scale predicted: the scale of the image used to predict. If half size was used (320x240) then value should be 0.5
	show_false_negatives_from_dir: If not None, then function will use the given directory to view images with false negtives highlighted """
	#Read data from csvs to memory
	predicted_objects = idputils.read_segmentation_csv(filename_predicted)
	true_objects   = idputils.read_segmentation_csv(filename_true)

	#Make sure all predicted images are labeled !
	print set(predicted_objects.keys())-set(true_objects.keys())
	if len(set(predicted_objects.keys())-set(true_objects.keys())) > 0:
		print 'Error: Not all predicted images are found in the labeled file. Exiting..'
		exit(1)
	
	if show_false_negatives_from_dir:
		imgname_tuples = idputils.list_images(show_false_negatives_from_dir) 
		imgname_by_prefix = {prefix:colorname for colorname,depthname,prefix in imgname_tuples}
	
	# NOW CALCULATE TOTAL SCORE
	total_score = 0
	count = 0
	false_positive = 0
	false_negative = 0
	for prefix in predicted_objects.keys():
		predicted_boxes = predicted_objects[prefix]
		true_boxes      = true_objects[prefix]
		
		while len(true_boxes) > 0 and len(predicted_boxes) > 0:
			max_accuracy = float('-inf')
			max_tidx = -1
			max_pidx = -1
			for t_idx, true_box in enumerate(true_boxes):
				for p_idx,predicted_box in enumerate(predicted_boxes):
					acc = get_box_accuracy(true_box, predicted_box, scale_predicted)
					if acc > max_accuracy:
						max_accuracy = acc
						max_tidx = t_idx
						max_pidx = p_idx
			
			count += 1
			del predicted_boxes[max_pidx]
			del true_boxes[max_tidx]
			total_score += max_accuracy
		
		false_positive += len(predicted_boxes)
		false_negative += len(true_boxes)
		
		if len(true_boxes) > 0 and show_false_negatives_from_dir: #VIEW FALSE NEGATIVES
			img = cv2.imread(imgname_by_prefix[prefix])
			for y1,x1,y2,x2 in true_boxes:
				idputils.red_rect(img, y1,x1,y2,x2)
			idputils.imshow(img, 'Check the FALSE NEGATIVES')

	print total_score
	total_score = total_score*1.0 / count #get average best_score
	print 'TOTAL:', total_score
	print 'FP:%i,  FN:%i' % (false_positive, false_negative)
	return total_score, false_positive, false_negative
	exit(0)
				
			
if __name__ == "__main__":
	
	parser = ArgumentParser()
	parser.add_argument('-p', dest='predicted_csv', help='Path to csv with predicted object segmentation')
	parser.add_argument('-l', dest='labeled_csv', help='Path to csv with Labeled object segmentation')
	parser.add_argument('-s', dest='scale', type=float, help='The scale at which coordinates of the predicted csv are in. Labeled csv is always assumed to be in scale 1 (for 640x480 images)')
	parser.add_argument('--show-false-negatives',dest='showfn', default=False, action = "store_true", help="Show images with False negative objects highlighted.")
	parser.add_argument('-i', dest='images_dir',default=None,help='If --show-false-negatives option is used, then -i is the directory containing the images (original scale: 640x480 images)')
	args = parser.parse_args()

	if (args.showfn and not args.images_dir) or (not args.showfn and args.images_dir):
		parser.error("-i and --show-false-negatives are either used together or not used at all.")
		
	get_file_accuracy(args.labeled_csv, args.predicted_csv, args.scale, args.images_dir)
	if len(sys.argv) < 3:
		print "Usage: python %s csv_predicted csv_labeled scale. Where csv_predicted is a filepath/name.\n Example python segmentation_error.py /tmp/file1.csv /tmp/file2.csv 0.3"
		exit(1) 
			
			
			
			