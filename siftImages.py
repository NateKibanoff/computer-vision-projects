# Import the necessary libraries
import cv2
import numpy as np
import sys

# Chi-squared distance to determine dissimilarity between two histograms
def chiSquared(h1, h2):
    x2 = 0
    for i in range(len(h1)):
        if h1[i]+h2[i] > 0: x2 += ((h1[i]-h2[i])**2)/(h1[i]+h2[i])
    return 0.5*x2

# Format the dissimilarity matrix (Task 2) to a more readable format
# All values are rounded off to two decimal places
# Parameters:
#   labels - list containing the image filenames
#   matrix - the computed dissimilarity matrix
def format_output(labels, matrix):
    longest = 4
    for i in labels: longest = max(longest, len(i[:-4]))
    space = "    "
    col_names = longest*" "
    for i in labels:
        col_names += space + (longest - len(i[:-4]))*" " + i[:-4]
    print(col_names)
    for i in range(len(matrix)):
        curr = labels[i][:-4]
        row_i = curr + (longest - len(curr))*" "
        for j in matrix[i]:
            row_i += space + (longest - 4)*" " + "%.2f"%j
        print(row_i)

# Task One: Display a given image and its luminance component with the keypoints
# The keypoints are marked with a '+' symbol, a circle representing the scale of the keypoint,
# And a line denoting the keypoint's orientation
# Parameters:
#   imgName - filename of the image to be displayed
#   display - determines if the images should be displayed or not (default True)
# Returns:
#   des - keypoint descriptors
def taskOne(imgName, display=True):
    # Read and rescale the image
    img = cv2.imread(imgName)
    scale = min(480/img.shape[0], 600/img.shape[1])
    new_dims = (int(img.shape[1]*scale), int(img.shape[0]*scale))
    img = cv2.resize(img, new_dims, interpolation = cv2.INTER_AREA)
    
    # Extract the luminance component from the image
    y_comp = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))[0]
    
    # Compute the image's keypoints using the SIFT algorithm
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(y_comp, None)
    print("Number of keypoints in", imgName, "is", des.shape[0])
    
    # Draw the keypoints according to the described specifications and display the results
    if display:
        y_keys = np.copy(img)
        y_keys = cv2.drawKeypoints(y_comp, kp, y_keys, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for i in kp: cv2.drawMarker(y_keys, (int(i.pt[0]), int(i.pt[1])), (0,0,0), markerSize = 5)
        cv2.imshow("Task 1", np.hstack((img,y_keys)))
        cv2.waitKey(0)

    # Return the list of keypoints
    return des

# Task two: Given a list of images, calculate and cluster all their keypoints
# using the k-means algorithm. After that, construct a histogram of keypoints
# per cluster for each image and compute their pairwise Chi-squared distance.
# The output should display a dissimilarity matrix for each pair of images
# Parameters:
#   imgList - a list of image filenames
def taskTwo(imgList):
    # Calculate the keypoint descriptors for each image in imgList
    imgs = None
    kp_i = []
    for i in range(len(imgList)):
        curr = taskOne(imgList[i], False)
        kp_i.append(curr.shape[0])
        if imgs is None: imgs = np.copy(curr)
        else: imgs = np.vstack((imgs, curr))
    print("Total number of keypoints is", len(imgs))
    
    # After computing the total number of keypoints of all images,
    # Perform the k-means algorithm with k = 5%, 10%, and 20% of
    # the total number of keypoints
    k_values = [5, 10, 20]
    for kv in k_values:
        k = int(0.01*kv*len(imgs))
        print("K = " + str(kv) + "%*(total number of keypoints) = " + str(k))
        print()
        print("Dissimilarity Matrix")
        print()
        
        # Peform the k-means clustering algorithm on the calculated keypoints
        # Sample implementation from https://github.com/opencv/opencv/blob/4.x/samples/python/kmeans.py
        clustered_kps = cv2.kmeans(imgs, k, None,
            (cv2.TERM_CRITERIA_EPS, 30, 0.1),
            10, 0)[1]
        
        # For each image, construct a histogram of keypoints per cluster
        hist = np.zeros((len(imgList), k))
        lastIdx = 0
        for i in range(len(imgList)):
            clusters = clustered_kps[lastIdx:lastIdx+kp_i[i]]
            lastIdx += kp_i[i]
            for j in clusters: hist[i][j] += 1
            hist[i]/=kp_i[i]
        
        # For each pair of images, compute their Chi-squared distance
        # and output the dissimilarity matrix with a readable format
        diss = np.zeros((len(imgList), len(imgList)))
        for i in range(len(diss)):
            for j in range(len(diss)):
                if j<i: diss[i][j] = diss[j][i]
                elif j>i: diss[i][j] = chiSquared(hist[i], hist[j])
        format_output(imgList, diss)
        if kv < 20:
            print()
            print()

if __name__ == '__main__':
    # If there's only one command line argument, perform Task One
    # If there are multiple command line arguments, perform Task Two
    if len(sys.argv) <= 2: taskOne(sys.argv[1])
    else: taskTwo(sys.argv[1:])