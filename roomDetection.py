# USAGE
# python roomDetection.py --image room.png

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import timeit

startTime = timeit.default_timer()

# start definate Point
class Point:
	def __init__(self, x, y, roomNum):
		self.x = x
		self.y = y
		self.roomNum = roomNum
	def __str__(self):
		return "[{}, {} | {}]".format(self.x, self.y, self.roomNum)
	def __repr__(self):
		return "[{}, {} | {}]".format(self.x, self.y, self.roomNum)
# end definate Point

def showAndStop(image):
	cv2.imshow("image", image)
	cv2.waitKey(0)
	return

def drawPoints(image, ptrs, bold):
	for p in ptrs:
		cv2.line(image, (p[0], p[1]), (p[0], p[1]),(255,0,0), bold)
	return image

def getImagePath():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to the input image")
	args = vars(ap.parse_args())
	return args["image"]

def printPythonObject(obj):
	print(obj)
	print(type(obj))
	print("object shape : {}".format(obj.shape))
	return

def printPointsInCounter(image, c):
	# start - draw the points in c
	printPythonObject(c)
	for element in c:
		print(element)
		for p in element: 
			print(p)
			print("{} - {}".format(p[0], p[1]))
			cv2.line(image, (p[0], p[1]), (p[0], p[1]),(255,0,0), 10)
	# end - draw the points in c
	return

def getCounters(image):
	# load the image, convert it to grayscale, blur it slightly,
	# and threshold it
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

	# showAndStop(image)

	realCnts = []
	# loop over the contours
	for c in cnts:

		# skip the small area
		perimeter = cv2.arcLength(c, True)
		if (perimeter < 100):
			continue

		realCnts.append(c)
	return realCnts

def getMaskImage(image, lower, upper):
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	# showAndStop(mask)

	return output

def getColoredPoints(image):
	# start - slow block
	ptrs = []
	for x_idx, xs in enumerate(image) :
		for y_idx, bpt in enumerate(xs) :
			# bpt is the RBG value for the current point
			if (bpt[0] + bpt[1] + bpt[2] > 0) : 
				ptrs.append([y_idx, x_idx])
	# end - slow block
	return ptrs


def buildYBasedModel(model, ptrs, num):
	for p in ptrs:
		pObj = Point(p[0], p[1], num)

		if (model[p[1]] == None):
			model[p[1]] = [pObj]
		else : 
			model[p[1]].append(pObj)

	return model

def buildXBasedModel(model, ptrs, num):
	for p in ptrs:
		pObj = Point(p[0], p[1], num)

		if (model[p[0]] == None):
			model[p[0]] = [pObj]
		else : 
			model[p[0]].append(pObj)

	return model

def drawModelPoints(image, model, color):
	for l in model:
		if l == None:
			continue
		for p in l:
			if p == None:
				continue
			cv2.line(image, (p.x, p.y), (p.x, p.y),color, 1)
	return

def draweRoomNum(image, cnts):
	i = 0
	for c in cnts:
		# start - draw solid circle and put text on the No.i center of the shape
		# compute the center of the contour
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])	

		cv2.circle(image, (cX, cY), 2, (255, 255, 255), -1)
		cv2.putText(image, "R{}".format(i), (cX - 10, cY - 10), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	 	# end - draw solid circle and put text on the center of the shape
		i=i+1
	return

def calculateLinkedRoom(image, model, modelTpye, topRel):

	# # start - test - Tony
	# xIdx = 90
	# yIdx = 625
	# drawPoints(image, [[xIdx, yIdx]], 10)
	# for l_idx, l in enumerate(model[xIdx:]):
	# end - test - Tony

	for l_idx, l in enumerate(model):
		if l == None:
			continue

		if modelTpye == 'Y':
			l = sorted(l, key=lambda p: p.x)
		else:
			l = sorted(l, key=lambda p: p.y)

		preP = l[0]

		# print("{} : {}".format(l_idx + xIdx, len(l)))
		# print(l)
		
		for p_idx, p in enumerate(l[1:]):
			if p == None:
				continue
			if preP.roomNum == p.roomNum:
				preP = p
				continue

			# print("preP {} -  p {}".format(preP, p))

			if modelTpye == 'Y':
				dist = abs(preP.x - p.x)
			else : 
				dist = abs(preP.y - p.y)

			if (dist < 10):
				if preP.roomNum > p.roomNum:
					topRel.add(str(preP.roomNum) + "_" + str(p.roomNum))
				else:
					topRel.add(str(p.roomNum) + "_" + str(preP.roomNum))

				# print("{} - {}".format(preP, p))
			
			preP = p	

	return topRel

def run():
	imagePath = getImagePath()
	image = cv2.imread(imagePath)
	cnts = getCounters(image)

	# start - detect the linked room base Y and X view
	yModel = [None] * 1000
	xModel = [None] * 1000
	# loop over the contours
	totalT = 0
	i = 0
	for c in cnts:

		imageCopy = image.copy()

		highlight = (255, 0, 0)
		lower = np.array([245, 0, 0])
		upper = np.array([255, 50, 50])

		# draw the contour of the shape on the image
		cv2.drawContours(imageCopy, [c], -1, highlight, 1)
		
		# showAndStop(imageCopy)

		maskedImg = getMaskImage(imageCopy, lower, upper)
		# showAndStop(imageCopy)	

		st = timeit.default_timer()
		ptrs = getColoredPoints(maskedImg)
		et = timeit.default_timer()
		print("time for get Points in Room{} : {}".format(i, et - st))
		totalT = totalT + (et - st)

		yModel = buildYBasedModel(yModel, ptrs, i)
		xModel = buildXBasedModel(xModel, ptrs, i)

		i = i + 1
		# if i > 7:
		# 	break
	
	print("totoal time for get points : {}".format(totalT))

	drawModelPoints(image, yModel, (0,255,0))
	draweRoomNum(image, cnts)

	topRel = set()
	topRel = calculateLinkedRoom(image, yModel, 'Y', topRel)
	topRel = calculateLinkedRoom(image, xModel, 'X', topRel)

	topRelList = list(topRel)
	topRelList = sorted(topRelList)
	print(topRelList)

	showAndStop(image)
	cv2.imwrite("img/room_out.png", image)

	# end - detect the linked room base Y and X view

	return

run()

endTime = timeit.default_timer()
print endTime - startTime






