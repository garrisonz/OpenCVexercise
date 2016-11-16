# USAGE
# python roomDetection.py --image room.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
 
# load the image, convert it to grayscale, blur it slightly,
# and threshold it
image = cv2.imread(args["image"])
kernel = np.ones((5,5),np.uint8)

# start - opening
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
image = opening
# end - opening

# start - erosion
erosion = cv2.erode(image,kernel,iterations = 1)
image = erosion
# end - erosion

# start - closing
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
image = closing
# End - closing


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 43, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

cv2.imshow("Image", image)
cv2.waitKey(0)

# loop over the contours
i = 0
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
 
	perimeter = cv2.arcLength(c, True)
	if (perimeter < 100):
		continue
		
	print("perimeter : {}".format(perimeter))

	# draw the contour and center of the shape on the image
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

	# start - draw solid circle and put text
	cv2.circle(image, (cX, cY), 2, (255, 255, 255), -1)
	cv2.putText(image, "R{}".format(i), (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 	# end - draw solid circle and put text

	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
	i=i+1
cv2.imwrite("img/room_out.png", image)


