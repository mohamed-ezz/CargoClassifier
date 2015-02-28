"""
This Standalone script calculates the get_box_accuracy of certain segmentation algorithm compared to ground truth(optimal) segmentation
The input is 2 segmentation csv files. Each csv has the same format as the following:
image_id/prefix , box_id, y1, x1, y2, x2

The get_box_accuracy is a number ranging from 0 to 1.0.
The number detected boxes may be different than the true number of objects (as per the labeling)
there could be more detected boxes than there is, or less.

The get_box_accuracy out of two files is calculated as follows :
For each image in the predicted: get get_box_accuracy of each predicted bounding box as the ratio of overlap area
over the total are of the 2 boxes (predicted and true). Best score is 1, worst is 0.
Then each extra box predicted, penalizes the score by -1 
And each missing box (not predicted), penalizes the score by -1.
"""
import sys
from collections import defaultdict

if len(sys.argv) < 3:
	print 'Usage: segmentation_error.py predicted.csv labeled.csv'
	exit(1)
	
def get_area(box):
	y1, x1, y2, x2 = box
	return abs((y2-y1)*(x2-x1))


def get_overlap(boxA,boxB):
	"""Get number of pixels of 2 rectangles, each defined by 2 points"""
	#Ensure that first point is top left of the second point for Box A
	ay1, ax1, ay2, ax2 = min(boxA[0],boxA[2]), min(boxA[1],boxA[3]), max(boxA[0],boxA[2]), max(boxA[1],boxA[3])
	#same for Box B
	by1, bx1, by2, bx2 = min(boxB[0],boxB[2]), min(boxB[1],boxB[3]), max(boxB[0],boxB[2]), max(boxB[1],boxB[3])
	
	x_overlap = max(0, min(ax2,bx2) - max(ax1,bx1))
	y_overlap = max(0, min(ay2,by2) - max(ay1,by1))
	overlap = x_overlap * y_overlap
	return overlap
	
	
	
def get_box_accuracy(true_box, predicted_box):
	tarea = get_area(true_box)
	parea = get_area(predicted_box)
	
	overlaparea = get_overlap(true_box, predicted_box)
	#print tarea,parea,overlaparea
	score = (overlaparea*2.0) / (tarea+parea)
	return score

def get_file_accuracy(filename_true, filename_predicted):
	
	predicted_file = open(filename_predicted,'r')
	labeled_file = open(filename_true,'r')
	
	
	#Read data from csvs to memory
	true_objects = defaultdict(list) # {img_prefix : [object, object, object...]} an object is : [y1,x1,y2,x2].
	for line in labeled_file:
		if line.startswith('image_id,'): continue #skip header
		prefix, box_id, y1, x1, y2, x2 = map(int,line.split(',')) #parse
		true_objects[prefix].append([y1,x1,y2,x2])
		
	predicted_objects = defaultdict(list)
	for line in predicted_file:
		if line.startswith('image_id,'): continue #skip header
		prefix, box_id, y1, x1, y2, x2 = map(int,line.split(',')) #parse
		predicted_objects[prefix].append([y1,x1,y2,x2])
		
	#Make sure all predicted images are labeled !
	if len(set(predicted_objects.keys())-set(true_objects.keys())) > 0:
		print 'Error: Not all predicted images are found in the labeled file. Exiting..'
		exit(1)
	
	# NOW CALCULATE TOTAL SCORE
	total_score = 0
	count = 0
	false_positive = 0
	false_negative = 0
	for prefix in predicted_objects.keys():
		predicted_boxes = predicted_objects[prefix]
		true_boxes      = true_objects[prefix]
		#print prefix, true_boxes
		
		for predicted_box in predicted_boxes:
			#Which box from true should correspond to the current predicted_box ? the one with highest score/get_box_accuracy
			max_accuracy = float('-inf') #get_box_accuracy between 2 boxes, predicted and true
			max_idx = -1
			for i,true_box in enumerate(true_boxes):
				acc = get_box_accuracy(true_box, predicted_box)
				if acc > max_accuracy:
					max_accuracy = acc
					max_idx = i
			
			#print len(true_boxes)
			#print max_accuracy,max_idx
			count += 1
			if max_idx == -1: #no corresponding box. Predicted more boxes than labeled !!!
				#total_score -= 1.0
				false_positive += 1
				continue
			
			total_score += max_accuracy
			del true_boxes[max_idx]		
		false_negative += len(true_boxes)
		#total_score -= len(true_boxes) * 1.0
		
		
	total_score = total_score*1.0 / count #get average score
	print 'TOTAL:', total_score
	print 'FP:%i,  FN:%i' % (false_positive, false_negative)
	return total_score, false_positive, false_negative
				
				
			
if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "Usage: python %s csv_predicted csv_labeled. Where csv_predicted is a filepath/name"
		exit(1) 
	get_file_accuracy(sys.argv[2], sys.argv[1])
			
			
			
			