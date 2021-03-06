DEPTH_HI = 3500
DEPTH_LO = 1000


# Depth cutoff --> Buildmatrix (x,y,z,hue) --> Kmeans --> Blur --> Seed search --> Region growing



#Segmentation parameters (every single parameter is included)
PREDICT_AT_SCALE = 0.4
MEDIAN_BLUR_KERNEL_SIZE = 45
# Y_KMEANS_SCALE = 3
# X_KMEANS_SCALE = 7
# DEPTH_KMEANS_SCALE = 10
# COLOR_KMEANS_SCALE = 3

Y_KMEANS_SCALE = 3.44295655
X_KMEANS_SCALE = 6.69883958
DEPTH_KMEANS_SCALE = 10.65144706
COLOR_KMEANS_SCALE = 4.26427488


# Number of seed sizes
# The preset seed sizes
SEED_HEIGHT= 100
SEED_WIDTH = 40
# Seed step size

#
# Scoring function (what it is)
# Weights of scoring function terms
AREA_WEIGHT = 12.47683959
NONBG_WEIGHT = 7.94501212
INVENTROPY_WEIGHT = 3.15075799
# AREA_WEIGHT = 5
# NONBG_WEIGHT = 3.5
# INVENTROPY_WEIGHT = 4


# NUMBER OF OBJECTS TO LOOK FOR
N_OBJECTS = 3
MIN_ACCEPTED_OBJECT_SCORE = 8


# LABEL TO STR
lbl_str = {0:'BARREL',1:'BLUEBOX',2:'BROWNBOX',3:'irrelevant'}
str_lbl = {v:k for k,v in lbl_str.items()}



N_PARALLEL_PROCESSES = 7