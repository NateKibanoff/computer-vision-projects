# Import the necessary libraries
import cv2
import numpy as np
import sys

arg = sys.argv[1] # take the command line arguments
commands = ["-XYZ", "-Lab", "-YCrCb", "-HSB"] # possible commands for Task One

# Task One: Display a given image file and its components of a specified color space
# Command line arguments: -XYZ|-Lab|-YCrCb|-HSB imagefile
# First command line argument represents the desired color space
# (CIE-XYZ, CIE-Lab, YCrCb and HSB respectively)
# Second command line argument represents the image filename
# If the first command line does not follow the required format, move to Task Two
if arg in commands:
    img = cv2.imread(sys.argv[2])
    c1,c2,c3 = None,None,None
    
    # The maximum size of the viewing window is 1280 x 720
    # The output is split into four quadrants
    # The following snippet reduces the image size to follow this requirement
    # The original aspect ratio is maintained
    if img.shape[1] > 640 or img.shape[0] > 360:
        scale = min(640/img.shape[1], 360/img.shape[0])
        new_dims = (int(img.shape[1]*scale), int(img.shape[0]*scale))
        img = cv2.resize(img, new_dims, interpolation = cv2.INTER_AREA)
        
    # Split the image's color components based on the specified color space
    # According to the instructions, if the selected color space is HSB,
    # the B component is on the top-right, the H component is on the bottom-left,
    # and the S component is on the bottom-right
    if arg == commands[0]: c1,c2,c3 = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2XYZ))
    elif arg == commands[1]: c1,c2,c3 = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    elif arg == commands[2]: c1,c2,c3 = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    else: c2,c3,c1 = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    
    # Splitting the image's color space components result into three images represented as 2D numpy arrays
    # We need to convert the resulting images into 3D arrays by representing each pixel with three identical values
    for k in range(3):
        temp = []
        for i in range(img.shape[0]):
            temp.append([])
            for j in range(img.shape[1]):
                if k==0: temp[i].append(3*[c1[i,j]])
                elif k==1: temp[i].append(3*[c2[i,j]])
                else: temp[i].append(3*[c3[i,j]])
        if k==0: c1 = np.array(temp)
        elif k==1: c2 = np.array(temp)
        else: c3 = np.array(temp)
        
    # Combine and display the resulting images
    top = np.concatenate((img,c1), axis=1)
    bottom = np.concatenate((c2,c3), axis=1)
    full = np.concatenate((top,bottom))
    cv2.imshow("Task 1", full)
    cv2.waitKey(0)
    
# Task Two: Perform chroma keying on a given scenic photo and a green screen photo
# (i.e. remove the green background from the green screen photo, and replace it with
# the scenic photo as the new background)
# First command line represents the filename for the intended background photo
# Second command line represents the filename for the image with a green screen background
else:
    scene = cv2.imread(arg)
    green = cv2.imread(sys.argv[2])
    shrunk = False
    
    # Shrink the background photo so the output has a maximum size of 1280 x 720
    if scene.shape[1] > 640 or scene.shape[0] > 360:
        scale = min(640/scene.shape[1], 360/scene.shape[0])
        new_dims = (int(scene.shape[1]*scale), int(scene.shape[0]*scale))
        scene = cv2.resize(scene, new_dims, interpolation = cv2.INTER_AREA)
    
    # Shrink or expand the green screen photo so that it has the same width as the background
    # This step is needed so that the output can be properly formatted
    if green.shape[0] > scene.shape[0] or green.shape[1] > scene.shape[1]:
        scale = min(scene.shape[1]/green.shape[1], scene.shape[0]/green.shape[0])
        new_dims = (scene.shape[1], int(green.shape[0]*scale))
        green = cv2.resize(green, new_dims, interpolation = cv2.INTER_AREA)
        shrunk = True
    if (green.shape[0] < scene.shape[0] or green.shape[1] < scene.shape[1]) and not shrunk:
        scale = max(scene.shape[1]/green.shape[1], scene.shape[0]/green.shape[0])
        new_dims = (scene.shape[1], int(green.shape[0]*scale))
        green = cv2.resize(green, new_dims, interpolation = cv2.INTER_AREA)
    
    # A pixel is considered to be part of the green screen background if A*(r + b) - B*g <= 0
    # Where r, g, b represents a pixel's RGB values and A and B are arbitrarily defined constants
    # In this implementation, A = 1.0 and B = 1.1
    # Formula derived from https://en.wikipedia.org/wiki/Chroma_key#Programming
    mask = []
    for i in range(green.shape[0]):
        mask.append([])
        for j in range(green.shape[1]):
            if int(green[i,j,0])+int(green[i,j,2])-1.1*int(green[i,j,1])<1: mask[i].append(255)
            else: mask[i].append(0)
    mask = np.array(mask, dtype=np.uint8)

    # Remove green screen background and replace it with a white background
    foreground = cv2.bitwise_and(green, green, mask = mask)
    black = green - foreground
    white = np.copy(black)
    white[mask == 255] = [255,255,255]
    
    # Replace green screen background with the provided scenic photo
    combined = np.copy(scene)
    combined[(scene.shape[0]-green.shape[0]):][mask == 0] = [0,0,0]
    combined[(scene.shape[0]-green.shape[0]):] += black
    
    # Combine and display the resulting images
    left = np.concatenate((green,scene))
    right = np.concatenate((white,combined))
    full = np.concatenate((left,right), axis=1)
    cv2.imshow("Task 2", full)
    cv2.waitKey(0)