# Description of the solution for tracking the three pedestrians
# who are closest to the camera: My solution assumes that the
# people who are at the bottom of the screen are closest to the
# camera. Following this assumption, my solution sorts the detected
# persons via their y-coordinates. The three persons with the
# highest values for the y-coordinate (i.e. bottom of the frame)
# will be highlighted and labelled with the bounding boxes

# Import the necessary libraries
import cv2
import numpy as np
import sys

# Helper function for padding an image and to make it into a square
# while still maintaining the original image's aspect ratio. This is
# used for Task Two because the pre-trained model assumes that the
# input images would have a resolution of 300 x 300
# Parameters:
#   img - the input image for the pre-trained model
# Returns:
#   img - the padded version of the input image
def padImage(img):
    pad = []
    
    # This function adds extra rows/columns of black pixels
    # until the image has an aspect ratio of 1:1; if the
    # input frame already has this aspect ratio, then no
    # changes are made
    if img.shape[0] > img.shape[1]: #height > width
        for i in range(img.shape[0]):
            pad.append([])
            for j in range(img.shape[0] - img.shape[1]): pad[i].append([0,0,0])
        img = np.hstack((img, np.array(pad,dtype=np.uint8)))
    if img.shape[1] > img.shape[0]: #width > height
        for i in range(img.shape[1] - img.shape[0]):
            pad.append([])
            for j in range(img.shape[1]): pad[i].append([0,0,0])
        img = np.vstack((img, np.array(pad,dtype=np.uint8)))
    return img

# Read the first frame of the input video file and compute its framerate
vidcap = cv2.VideoCapture(sys.argv[2])
success, frame = vidcap.read()
fps = vidcap.get(cv2.CAP_PROP_FPS)

# Rescale the first frame to a size comparable to VGA
# while maintaining the original aspect ratio
scale = min(600/frame.shape[1], 480/frame.shape[0])
new_dims = (int(frame.shape[1]*scale), int(frame.shape[0]*scale))

# Task One: Given a video file as input, display the original frame,
# the estimated background, a binary mask of the detected moving pixels
# (before noise removal), and the moving objects in original color (with
# a black background). We use the Gaussian mixture background modelling
# method to perform those tasks. Furthermore, we then use connected component
# analysis to classify the moving objects (person, car, others) using the ratio
# of each connected component's dimensions
if sys.argv[1] == "-b":
    # Initialize OpenCV's built-in feature for background subtraction
    backSub = cv2.createBackgroundSubtractorMOG2()
    
    frameCount = 1
    while success:
        # Resize each video frame
        frame = cv2.resize(frame, new_dims, interpolation = cv2.INTER_AREA)
        
        # Update the background model and generate the foreground mask
        fgMask = backSub.apply(frame)
        temp = []
        for i in range(fgMask.shape[0]):
            temp.append([])
            for j in range(fgMask.shape[1]):
                if fgMask[i,j]==255: temp[i].append([255,255,255])
                else: temp[i].append([0,0,0])
        binMask = np.array(temp, dtype=np.uint8)
        
        # the getBackgroundImage() method generates the estimated background
        top = np.hstack((frame, backSub.getBackgroundImage()))
        
        # noise removal via morphological operators using a 5x5 rectangular
        # structuring element (https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        filtered = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        moving = np.copy(frame)
        moving[filtered < 255] = [0,0,0]
        bottom = np.hstack((binMask, moving))
        
        # Display the images according to the assignment specificaions
        # The user can terminate the program by pressing the 'Esc' key
        cv2.imshow("Task One", np.vstack((top, bottom)))
        if cv2.waitKey(1) == 27: break
        
        # For object classification, we use OpenCV's built-in connected component analysis implementation
        # The first frame is always assumed to have 0 moving objects
        # https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
        if frameCount > 1:
            comps = cv2.connectedComponentsWithStats(filtered)
            text = "Frame " + str(frameCount) + ": " + str(comps[0]-1) + " object/s"
            if comps[0] > 1:
                text += " ("
                persons = 0
                cars = 0
                others = 0
                
                # In this implementation, we assume that each connected component is one object
                # If width/height >= 1.3, the object is assumed to be a car
                # If height/width >= 1.3, the object is assumed to be a person
                # If both conditions are false, the object is assumed to be some other object
                for i in range(1, comps[0]):
                    if comps[2][i, cv2.CC_STAT_WIDTH]/comps[2][i, cv2.CC_STAT_HEIGHT] >= 1.3: cars+=1
                    elif comps[2][i, cv2.CC_STAT_HEIGHT]/comps[2][i, cv2.CC_STAT_WIDTH] >= 1.3: persons+=1
                    else: others += 1
                text += str(persons) + " person/s, " + str(cars) + " car/s, " + str(others) + " other/s)"
            print(text)
        else: print("Frame 1: 0 object/s")
        frameCount += 1
        
        # Read the next frame (if any) and perform the same procedure
        success, frame = vidcap.read()

# Task Two
elif sys.argv[1] == "-d":
    # Load the pre-trained model
    model = cv2.dnn.readNet(model='frozen_inference_graph.pb',
        config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',framework='TensorFlow')
        
    # Counter for unique detected pedestrians
    coors = []
    unique = 0
    frameCount = 0
    
    while success:
        # Resize and pad the image. Use the padded image as input for the model
        frame = padImage(cv2.resize(frame, new_dims, interpolation = cv2.INTER_AREA))
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        model.setInput(blob)
        output = model.forward()
        
        # Draw a box around the detected pedestrians based on their locations and
        # respective confidence scores (50% threshold for this implementation)
        persons = []
        ur = np.copy(frame)
        temp = []
        for person in output[0,0,:,:]:
            if int(person[1]) == 1 and person[2] > 0.5:
                box_x = int(person[3]*frame.shape[1])
                box_y = int(person[4]*frame.shape[0])
                width = int(person[5]*frame.shape[1])
                height = int(person[6]*frame.shape[0])
                persons.append((box_x,box_y,width,height)) # record the upper-left coordinates of the bounding boxes
                cv2.rectangle(ur, (box_x,box_y), (width,height), (127,0,0), thickness=2)
                
                # Detected pedestrians from consecutive frames are assumed to be the same person
                # if their distance is close enough to each other
                if len(coors) > 1:
                    closest = len(coors)
                    dist = 999
                    for i in range(len(coors)):
                        if abs(box_x - coors[i][0]) <= 25 and abs(box_y - coors[i][1]) <= 25:
                            if np.sqrt((box_x - coors[i][0])**2 + (box_y - coors[i][1])**2) < dist:
                                dist = np.sqrt((box_x - coors[i][0])**2 + (box_y - coors[i][1])**2)
                                closest = i
                    if closest < len(coors):
                        coors[closest] = [box_x, box_y, width, height, coors[closest][4]]
                        temp.append(coors[closest])
                        coors.remove(coors[closest])
                    else:
                        unique = unique%99 + 1
                        temp.append([box_x, box_y, width, height, unique])
                else:
                    unique = unique%99 + 1
                    temp.append([box_x, box_y, width, height, unique])
        
        ll = np.copy(ur)
        lr = np.copy(frame)
        
        # Sort each person based on the y-coordinates. This is for pedestrian tracking and for
        # identifying the closest pedestrians to the camera. The closest pedestrian will be
        # labelled '1', the next closest will be labelled '2', etc.
        persons = np.array(persons, dtype = [("x",int),("y",int),("width",int),("height",int)])
        persons = np.sort(persons, order="y")
        for i in range(1,min(4,len(persons)+1)):
            box = persons[-i]
            cv2.rectangle(lr, (box[0],box[1]), (box[2],box[3]), (127,0,0), thickness=2)
            cv2.putText(lr, str(i), (box[0],box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,0), 2)
        
        # Tracking of all detected pedestrians
        for i in range(len(temp)):
            box = temp[i]
            cv2.rectangle(ll, (box[0],box[1]), (box[2],box[3]), (127,0,0), thickness=2)
            cv2.putText(ll, str(box[4]), (box[0],box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127,0,0), 1)
        
        # Display the images according to the assignment specificaions
        # The user can terminate the program by pressing the 'Esc' key
        # Afterwards, ead the next frame (if any) and perform the same procedure
        top = np.hstack((frame[:new_dims[1],:new_dims[0]], ur[:new_dims[1],:new_dims[0]]))
        bottom = np.hstack((ll[:new_dims[1],:new_dims[0]], lr[:new_dims[1],:new_dims[0]]))
        cv2.imshow("Task Two", np.vstack((top,bottom)))
        if cv2.waitKey(1) == 27: break
        success, frame = vidcap.read()
        
        # If a person has not been detected for at least two seconds, it is assumed
        # that they have exited the scene
        frameCount += 1
        if frameCount%(2*int(fps)) > 0: coors += temp
        else: coors = temp.copy()