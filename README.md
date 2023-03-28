# Computer Vision Projects
I took a computer vision algorithms and systems class last year and here are some of the assignments I made. To run these projects, we need the latest versions of Python, OpenCV, and numpy.

**TO DO: Add sample screenshots to the README file**

## chromaKey.py
### Submitted: 19 August 2022
#### Task One
Given an input image and the desired color space, split and display the image's three components according to the specified color space.

Supported color spaces: CIE-XYZ, CIE-Lab, YCrCb and HSV

Console command: `python chromaKey.py -XYZ|-Lab|-YCrCr|-HSB imageFile`
#### Task Two
Perform chroma keying on a given scenic photo and a green screen photo (i.e. remove the green background from the green screen photo, and replace it with the scenic photo as the new background)

Console command: `python chromaKey.py scenicImage greenScreenImage`

## siftImages.py
### Submitted: 17 September 2022
This assignment uses the [SIFT (scale-invariant feature transform) algorithm](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf) to detect an image's keypoints which will then serve as features in computing similarities between pairs of images.

#### Task One
Display a given image and its luminance component with the keypoints. The keypoints are marked with a '+' symbol, a circle representing the scale of the keypoint, and a line denoting the keypoint's orientation.

Console command: `python siftImages.py imageFile`
#### Task Two
Given a list of images, calculate and cluster all their keypoints using the [k-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering). After that, construct a histogram of keypoints per cluster for each image and compute their pairwise Chi-squared distance. The output should display a dissimilarity matrix for each pair of images

Console command: `python siftImages.py imageFile1 imageFile2 ...`

## movingObj.py
### Submitted: 30 October 2022

#### Task One
Given a video file as input, display the original frame, the estimated background, a binary mask of the detected moving pixels (before noise removal), and the moving objects in original color (with a black background). We use the Gaussian mixture background modelling method to perform those tasks. Furthermore, we then use connected component analysis to classify the moving objects (person, car, others) using the ratio of each connected component's dimensions.

Console command: `python movingObj.py -b videoFile`

#### Task Two

Given an input video file, detect and track all the pedestrians seen in the video. The program should display four screens:
- the original video frame
- video frame with overlapped bounding boxes of detected pedestrians
- video frame with labelled bounding boxes
- video frame where only three pedestrians who are closest to the camera are detected and tracked

This program makes use of a MobileNet Single Shot Detector (SSD) trained on the MS COCO dataset. The model weights and configuration are found in these files:
- frozen_inference_graph.pb
- ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt

A tutorial on how to use OpenCV's DNN module on the provided pre-trained model can be found [here](https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/).

Console command: `python movingObj.py -b videoFile`
