# CargoClassifier
This is an app that will be embedded in Fork lifts with the help of a Kinect camera mounted on the front of the fork lift.
The app notifies the driver of the type of cargo that is on the pallete right infront of the fork lift.

The scope of this codebase is to implement all necessary Computer Vision and Machine Learning algorithms for such a task. It involves object segmentation and classification. 

The project is structured in many scripts that take command line arguments. 

## Important files : 
1. idputils.py : contains all the common utility functions
2. extraction/ : Directory with segmentation and machine learninig scripts, other scripts under the root directory perform tasks like dataset augmentation/labeling tool/error calculation..etc.
3. extraction/kmeans.py : Main script for segmentation
4. extraction/learner.py : Main script for building a machine learning model
5. extraction/app.py : The Main Script. Executes the full pipeline on a dataset
