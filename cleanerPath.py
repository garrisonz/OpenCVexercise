# USAGE
# python cleanerPath.py --image singleRoom.png

# ID 		Developer	comment
# TZ1202_1	Tony		fix incorrect path
# TZ1202_2	Tony		Draw path from top to down, from left to right


# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
from itertools import izip
import time

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return izip(*[iter(iterable)]*n)

def drawPoints(image, ptrs, bold):
	for p in ptrs:
		cv2.line(image, (p[0], p[1]), (p[0], p[1]),(255,0,0), bold)
	return image

def showAndStop(image):
	cv2.imshow("image", image)
	cv2.waitKey(0)
	return

def getImagePath():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to the input image")
	args = vars(ap.parse_args())

	return args["image"]

def drawBoundaryPoints(imagePath):
	# load the image, convert it to grayscale, blur it slightly,
	# and threshold it
	image = cv2.imread(imagePath)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

	# find contours in the thresholded image
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# loop over the contours
	for c in cnts:
		# ignore small noise
		perimeter = cv2.arcLength(c, True)
		if (perimeter < 100):
			continue

		# draw the contour of the shape on the image
		cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
	return image

def getAllBoundaryPoints(image):
	lower = np.array([0, 245, 0])
	upper = np.array([50, 255, 50])

	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)

	print("output.shape : {}".format(output.shape))

	boundary_point = []
	for x_idx, xs in enumerate(output) :
		for y_idx, bpt in enumerate(xs) :
			# bpt is the RBG value for the current point
			if (bpt[0] + bpt[1] + bpt[2] > 0) : 
				boundary_point.append([y_idx, x_idx])
	# print(boundary_point)
	return boundary_point

def restoreBoundaryPointBaseY(boundary_point):
	PointBaseY = [None] * 500
	for p in boundary_point : 
		if (PointBaseY[p[1]] == None) :
			PointBaseY[p[1]] = [[p[0], p[1]]]
		else :
			PointBaseY[p[1]].append([p[0], p[1]])
	return PointBaseY

def getAllUncatenatedPoints(image, PointBaseY):
	uncatP = [None] * 500
	for l_i, l in enumerate(PointBaseY) : 
		if (l == None) :
			continue

		newList = []
		idx = 0
		# print(" oldArr[{}] : {}".format(l_i, l))
		while idx < len(l) :
			# find the left point of an path
			left = l[idx]
			nextIdx = idx + 1;
			if (nextIdx < len(l)) : 
				while (nextIdx < len(l) and (l[nextIdx][0] - left[0]) < 2 ) :
					left = l[nextIdx]
					nextIdx = nextIdx + 1
			
			# find the right point of an path
			if nextIdx < len(l) :
				right = l[nextIdx]

				# start TZ1202_1
				isValid = testIfValidPath(image, left, right)
				if isValid == False:
					idx = idx + 1
					continue
				# end TZ1202_1

				newList.append(left)
				newList.append(right)

				# skip all points catenating with the right point
				right_tmp = right
				nextIdx = nextIdx + 1
				if ( nextIdx < len(l)) : 
					while nextIdx < len(l) and l[nextIdx][0] - right_tmp[0] < 2:
						right_tmp = l[nextIdx]
						nextIdx = nextIdx + 1
			idx = nextIdx
		# print("uncatP[{}] : {}".format(l_i, newList))

		uncatP[l_i] = newList

		# showAndStop(image)

	return uncatP

def getPointBase10Y(PointsBaseY):
	points10Y = [[]] * (len(PointsBaseY) + 1 )
	totalA10B = 0
	i = 0
	j = 0
	while i < len(PointsBaseY):
		# print(" PointsBaseY[{}] : {}".format(i, PointsBaseY[i]))
		points10Y[j] = PointsBaseY[i]
		if (points10Y[j] != None) : 
			totalA10B = totalA10B + len(points10Y[j])
		i = i + 10
		j = j + 1
	return points10Y

# start TZ1202_1
def testIfValidPath(image, pl, pr):

	print("{}, {}".format(pl, pr))

	yaxis = pl[1]
	xc = (pl[0] + pr[0]) / 2
	xl = (pl[0] + xc) / 2
	xr = (xc + pr[0]) / 2

	rgbl = int(image[yaxis][xl][0]) + int(image[yaxis][xl][1]) + int(image[yaxis][xl][2])
	rgbc = int(image[yaxis][xc][0]) + int(image[yaxis][xc][1]) + int(image[yaxis][xc][2])
	rgbr = int(image[yaxis][xr][0]) + int(image[yaxis][xr][1]) + int(image[yaxis][xr][2])

	# drawPoints(image, [[xl, yaxis], [xc, yaxis], [xr, yaxis]], 4)

	# 255 * 3 is 765
	if rgbl < 765 and rgbc < 765 and rgbr < 765:
		return False

	return True
#ene TZ1201_1

def drawPath(image, pointBaseY):
	totalA10B = 0
	for l in pointBaseY:
		if (l != None):
			totalA10B = totalA10B + len(l)

	while totalA10B > 0:

		preRightP = None # TZ1202_2

		rightDown = True
		count = 0
		while count < len(pointBaseY) :
			if (pointBaseY[count] == None or len(pointBaseY[count]) == 0) : 
				count = count + 1
				continue
			lst = pointBaseY[count]
			
			# print("lst : {} -  {}".format(lst, count))
			leftP = lst[0]
			rightP = lst[1]

			# start TZ1202_2
			if preRightP == None:
				preRightP = rightP
			elif preRightP[0] < leftP[0]:
				# draw from top to down again if the left point is righter 
				# than the previous right point
				count = count + 1
				continue
			else:
				preRightP = rightP
			# end TZ1202_2

			cv2.line(image, (leftP[0], leftP[1]), (rightP[0], rightP[1]),(255,0,0),1)
			lst.remove(leftP)
			lst.remove(rightP)
			totalA10B = totalA10B - 2
			# print("lst after remove : {} -  {}".format(lst, count))

			showAndStop(image)
			# cv2.imshow("image", image)
			# time.sleep(0.1)

			# start draw downline
			nextC = count + 1
			if (nextC < len(pointBaseY) and pointBaseY[nextC] != None and len(pointBaseY[nextC]) > 0):
				nextLst = pointBaseY[nextC]
				nextLP = nextLst[0]
				nextRP = nextLst[1]

				# start connect area test
				if (nextLP[0] > rightP[0]):
					continue
				# end connect area test

				if rightDown :
					cv2.line(image, (rightP[0], rightP[1]), (nextRP[0], nextRP[1]),(255,0,0),1)
				else :
					cv2.line(image, (leftP[0], leftP[1]), (nextLP[0], nextLP[1]),(255,0,0),1)
				rightDown = False if rightDown else True
			# end draw downline
			count = count + 1 # path gap
	return image

def printYBasedPoint(YBasePoints):
	for l in YBasePoints:
		if (l == None):
			continue
		print("["),
		for p in l :
			print(p),
		print("]")
	return

def highlineYBasePoint(YBasePoints, img, bold):
	for l in YBasePoints:
		if (l == None) or len(l) == 0:
			continue
		print("["),
		for p in l :
			print(p),
			cv2.line(img, (p[0], p[1]), (p[0], p[1]),(255,0,0), bold)
		print("]")
	return img

def run():
	imagePath = getImagePath()
	image = drawBoundaryPoints(imagePath)

	BoundaryPoints = getAllBoundaryPoints(image)

	boundaryPointsBaseY = restoreBoundaryPointBaseY(BoundaryPoints)

	# printYBasedPoint(boundaryPointsBaseY)
	# image = highlineYBasePoint(boundaryPointsBaseY, image, 1)
	# showAndStop(image)
	# return

	uncatenatePoint = getAllUncatenatedPoints(image, boundaryPointsBaseY)

	# printYBasedPoint(uncatenatePoint)
	# image = highlineYBasePoint(uncatenatePoint, image, 4)
	# showAndStop(image)
	# return


	pointBase10Y = getPointBase10Y(uncatenatePoint)

	# image = highlineYBasePoint(pointBase10Y, image, 4)
	# showAndStop(image)
	# return

	image = drawPath(image, pointBase10Y)

	# show the images
	cv2.imshow("image", image)
	cv2.waitKey(0)
	cv2.imwrite("img/singleRoom_out.png", image)
	return

run()



