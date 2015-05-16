txt="""
Takes a segmentation_label.csv file (which have object bounding box locations for each image, with each object as one row)
and appends a new column to the csv with the object-type label (0=BARREL, 1=BLUE, 2=BROWN)

It knows the type of object by looking into a directory with all the objects separated into folders named BARREL,BLUEBOX,BROWNBOX,irrelevant
so for each row of the segmentation_label.csv, it looks into the 4 directories for the current row/img, 
and puts the label according to which folder containing it

Example:
python append_object_label.py -l /idpdata/frontal/segmentation_labels.csv -i /idpdata/frontal_labeledobjects/orig
"""
import idputils
from argparse import ArgumentParser
import os,glob


if __name__=='__main__':
	parser = ArgumentParser(description=txt)
	parser.add_argument('-l', dest='labeled_csv', required=True, help='Path to csv with Labeled object segmentation')
	parser.add_argument('-i', dest='objects_dir', required=True, help='A folder that contains 3 folders BARREL,BLUEBOX,BROWNBOX, representing the object-type label')
	parser.add_argument('-o', dest='output_path')
	args = parser.parse_args()

	barrelpath = os.path.join(args.objects_dir,  'BARREL/')
	bluepath   = os.path.join(args.objects_dir,  'BLUEBOX/')
	brownpath  = os.path.join(args.objects_dir,  'BROWNBOX/')
	
	file = open(args.labeled_csv,'r')
	headerstr = file.readline()
	#outfile = open(args.output_path, 'w')
	#outfile.write(headerstr)
	print headerstr.strip()+',objectlabel'
	for line in file:
		prefix, boxid, y1,x1,y2,x2 = line.split(',')[:6]
		inbarrel    = len(glob.glob(barrelpath + prefix +'_'+ boxid + '_*')) > 0
		inblue  = len(glob.glob(bluepath + prefix +'_'+ boxid + '_*')) > 0
		inbrown = len(glob.glob(brownpath + prefix +'_'+ boxid + '_*')) > 0
		label       = '0' if inbarrel else '1' if inblue else '2' if inbrown else 'UNKNOWN'
		line = line.strip() + ','+label
		print line
	exit(0)
		
	prefix_boxes = idputils.read_segmentation_csv(args.labeled_csv) #dict {prefix: [box1, box2]}
	for prefix, boxes in prefix_boxes.items():
		for i in range(len(boxes)):
			inbarrel    = len(glob.glob(barrelpath + prefix +'_'+ str(i+1) + '_*')) > 0
			inblue  = len(glob.glob(bluepath + prefix +'_'+ str(i+1) + '_*')) > 0
			inbrown = len(glob.glob(brownpath + prefix +'_'+ str(i+1) + '_*')) > 0
			label       = '0' if inbarrel else '1' if inblue else '2' if inbrown else 'UNKNOWN'
			
			y1,x1,y2,x2 = boxes[i]
			y1,x1,y2,x2 = str(y1), str(x1), str(y2), str(x2)
			
			rowstr = ','.join([prefix, str(i+1), y1, x1, y2, x2, label])
			print rowstr
			#outfile.write()
			#print prefix+'_'+ str(i+1), inbarrel, inblue, inbrown
		
		
		
		
		
		
		
		
		
		
		
		
		