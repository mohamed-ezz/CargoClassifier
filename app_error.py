"""
Calculates the error of the complete pipeline 

Example:
python segmentation_error.py -p /idpdata/seg_output/kmeans_growth/segmentation_output.csv -l /idpdata/frontal/segmentation_labels.csv --show-false-negatives -i /idpdata/frontal/ -s 0.4
"""
from collections import defaultdict
import idputils
from argparse import ArgumentParser
import numpy as np
import extraction.config as cfg

def convert_to_objcounts(prefix_object_dict):
	"""Converts from the output of idputils.read_segmentation_withobjecttype_csv {prefix:[(y1,x1,y2,x2,type),(),..etc]}
	to {prefix: (barrelcount, bluecount, browncount), prefix:...etc}"""
	
	res = defaultdict(lambda: np.array([0,0,0])) #return by default 0 counts
	for prefix in prefix_object_dict:
		counts = np.array([0,0,0])
		for _,_,_,_,objtype in prefix_object_dict[prefix]:
			objtype = cfg.str_lbl[objtype] #convert to numeric lbl
			counts[objtype] += 1
			
		res[prefix] = counts
		
	return res
	
def get_app_accuracy(filename_true, filename_predicted, scale_predicted, show_false_negatives_from_dir=None):
	"""Scale predicted: the scale of the image used to predict. If half size was used (320x240) then value should be 0.5
	show_false_negatives_from_dir: If not None, then function will use the given directory to view images with false negtives highlighted """
	#Read data from csvs to memory
	predicted_objects = convert_to_objcounts(idputils.read_segmentation_withobjecttype_csv(filename_predicted))
	true_objects   = convert_to_objcounts(idputils.read_segmentation_withobjecttype_csv(filename_true))
	
	print 'Predicted file has %i images mentioned.' % len(predicted_objects.keys())
	print 'Labeled file has %i images mentioned.' % len(true_objects.keys())			

	total_fp      = np.array([0,0,0])
	total_fn      = np.array([0,0,0])
	total_objects = np.array([0,0,0])
	for prefix in set(predicted_objects.keys() + true_objects.keys()):
		predicted_counts = predicted_objects[prefix]
		true_counts = true_objects[prefix] #returns np.array([0,2,1]) counts of barrel,blue,brown
		
		fp_count = np.clip(predicted_counts - true_counts, 0, 999) #clip -ve numbers to 0
		fn_count = np.clip(true_counts - predicted_counts, 0, 999)
	
		total_objects += true_counts
		total_fp+=fp_count
		total_fn+=fn_count
	
	
	print 'FP',total_fp,'FN',total_fn
	print 'Obj count',total_objects	
	
if __name__ == "__main__":
	
	parser = ArgumentParser()
	parser.add_argument('-p', dest='appoutput_csv', help='Path to csv with predicted object segmentation')
	parser.add_argument('-l', dest='labeled_csv', help='Path to csv with Labeled object segmentation')
	parser.add_argument('-s', dest='scale', type=float, help='The scale at which coordinates of the predicted csv are in. Labeled csv is always assumed to be in scale 1 (for 640x480 images)')
	args = parser.parse_args()

	print get_app_accuracy(args.labeled_csv, args.appoutput_csv, args.scale)			
			
			
			