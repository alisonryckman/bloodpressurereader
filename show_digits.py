import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
from transform import four_point_transform
from sorting_contours import alisonsort
import glob

widthmin = 5
widthmax = 40
heightmin = 30
heightmax = 50

def dilate_rows(thresh, y1, y2):
    roi = thresh[y1:y2, 0: thresh.shape[1]]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 4))
    roi = cv2.dilate(roi, kernel, iterations=1) 
    thresh[y1:y2, 0: thresh.shape[1]] = roi

def main():
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 1, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }
    #files = glob.glob("s51080/*.jpg")
    files = glob.glob("s5/*.jpg")
    for file in files:
        print("Processing %s" % file)
        image = cv2.imread(file)
        image = imutils.resize(image, height=700)
        #cv2.imshow("Image", image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200, 255)

        # find contours in the edge map, then sort them by their
        # size in descending order
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        displayCnt = None
        #print('these are the cnts' , cnts)
        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if the contour has four vertices, then we have found
            # the thermostat display
            if len(approx) == 4:
                displayCnt = approx
                (x, y, w, h) = cv2.boundingRect(c)
                #cv2.drawContours(image, displayCnt, 3, (0, 255, 0), 50))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break
        # extract the thermostat display, apply a perspective transform to it
        warped = four_point_transform(gray, displayCnt.reshape(4, 2))

        # threshold the warped image, then apply a series of morphological
        # operations to cleanup the thresholded image
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        try:
            dilate_rows(thresh, 5, 35)
            dilate_rows(thresh, 60, 90)
        except:
            continue

        segment_thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
        # find contours in the thresholded image, then initialize the
        # digit contours lists
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        digitCnts = []
        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit
            if h < heightmin or h > heightmax or w < widthmin or w > widthmax or x < 20 or y > 80:
                continue
            digitCnts.append(c)
            #print(x,y,w,h)
            #cv2.rectangle(warped, (x,y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.imshow("Warped", warped)
            #cv2.waitKey(0)

        # sort the contours from left-to-right, then initialize the
        # actual digits themselves
        try:
            digitCnts = alisonsort(digitCnts)[0]
        except:
            continue
        #digitCnts = sort_contours(digitCnts, method="left-to-right")[0]
        digits = []

        #print('digitCnts ', len(digitCnts))
        # loop over each of the digits
        for n,c in enumerate(digitCnts):
            (x, y, w, h) = cv2.boundingRect(c)
            roi = thresh[y:y + h, x:x + w]
            #print(x,y,w,h)
            # compute the width and height of each of the 7 segments
            # we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            dHC = int(roiH * 0.05)
            # define the set of 7 segments
            segments = [
                ((0, 0), (w, dH)),  # top
                ((0, 0), (dW, h // 2)), # top-left
                ((w - dW, 0), (w, h // 2)), # top-right
                ((0, (h // 2) - dHC), (w, (h // 2) + dHC)), # center
                ((0, h // 2), (dW, h)), # bottom-left
                ((w - dW, h // 2), (w, h)), # bottom-right
                ((0, h - dH), (w, h))   # bottom
            ]
            #for i in segments:
            #    cv2.rectangle(thresh, (i[0][0],i[0][1]), (500,300), (0, 255, 0), 2)
            on = [0] * len(segments)
            totals = [0.] * len(segments)

		# loop over the segments
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                cv2.rectangle(segment_thresh, (x + xA,y + yA), (x + xB, y + yB), (0, 255, 0), 1)
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                #print('area' ,area)
                # if the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                #print ('section', str(i), 'is', str(total/float(area)), 'percent filled')
                if area == 0:
                    pass
                else:
                    if total / float(area) > 0.45:
                        on[i] = 1
                    totals[i] = total / float(area) 
				# lookup the digit and draw it on the image

            # 1 is special
            if w < 15:
                on = (0, 0, 1, 0, 0, 1, 0)

            if tuple(on) in DIGITS_LOOKUP:
                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(str(digit))
            else:
                digits.append('x')

            cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # display the digits
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

        # put threshold image on original image
        thresh_rgb = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
        image[0: thresh_rgb.shape[0], 0: thresh_rgb.shape[1]] = thresh_rgb

        if len(digits) > 4:
            cv2.putText(image, "".join(digits[0:3]),(119,151), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv2.CV_AA)
            bnumber = "".join(digits[3:])
            if len(bnumber) == 2:
                bnumber = " " + bnumber
            cv2.putText(image, bnumber,(119,181), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.CV_AA)
        else:
            cv2.putText(image, "".join(digits),(119,151), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.CV_AA)
        cv2.imshow("Image", image)
        cv2.imshow("Segments", imutils.resize(segment_thresh, height=700))
        cv2.waitKey(0)
        
if __name__ == "__main__":
    main()
