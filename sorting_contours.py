# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import copy

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


def alisonsort(cnts):
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key = lambda b: b[1][1]))
	boundingBoxesspare = copy.deepcopy(boundingBoxes)
	boundingBoxes = list(boundingBoxes)
	for n,i in enumerate(boundingBoxes):
		#print(i)
		boundingBoxes[n] = list(i)
	#print(boundingBoxes)
	basecomparison = boundingBoxes[0][1]
	#print('this is the base comparison' , basecomparison)
	for i in boundingBoxes:
		if ((i[1] - basecomparison) > -5) and ((i[1] - basecomparison) < 5):
			i[1] = basecomparison
		else:
			basecomparison = i[1]
	#print('boundingbox[1][1]', type(boundingBoxes))
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key = lambda b: (b[1][1], b[1][0])))
	return (cnts, boundingBoxes)
