import cv2
import sys
import numpy as np
import datetime
from imutils.object_detection import non_max_suppression

image_path=''

try:
	if len(sys.argv) < 2:
		while not image_path:
			image_path=raw_input("Enter image file path: ")
	else:
		image_path=sys.argv[1]
	
	bCascade_profile = cv2.CascadeClassifier('cascade_lbp.xml')
	
	image=cv2.imread(image_path)
	start = datetime.datetime.now()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	bikes = bCascade_profile.detectMultiScale(
	       	gray,
	       	scaleFactor=1.007,
	       	minNeighbors=1,
			minSize=(50,100),
	       	flags=cv2.CASCADE_SCALE_IMAGE
	)
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bikes])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.15)
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	file_ = open("recog_time.txt","a") 
	file_.write("{}".format((datetime.datetime.now() - start).total_seconds())) 
	file_.write("\n") 
	file_.close()

	print("[INFO] detection took: {}s".format((datetime.datetime.now()-start).total_seconds()))
	cv2.imshow('Result', image)

except cv2.error:
	print "\nError. Check file path, cascade path or read OpenCV error message.\n"
	sys.exit(1)

key=cv2.waitKey(1) & 0xFF
cv2.waitKey()
cv2.destroyAllWindows()
