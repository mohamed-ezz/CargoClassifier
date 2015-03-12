DEPTH_HI = 3500
DEPTH_LO = 1000


# Depth cutoff --> Buildmatrix (x,y,z,hue) --> Kmeans --> Blur --> Seed search --> Region growing



#Segmentation parameters (every single parameter is included)
PREDICT_AT_SCALE = 0.4
MEDIAN_BLUR_KERNEL_SIZE = 45
Y_KMEANS_SCALE = 3
X_KMEANS_SCALE = 7
DEPTH_KMEANS_SCALE = 10
COLOR_KMEANS_SCALE = 3

# Number of seed sizes
# The preset seed sizes
SEED_HEIGHT= 50
SEED_WIDTH = 20
# Seed step size

#
# Scoring function (what it is)
# Weights of scoring function terms
AREA_WEIGHT = 5
NONBG_WEIGHT = 3.5
INVENTROPY_WEIGHT = 4

# EXECUTION
N_PARALLEL_PROCESSES = 18