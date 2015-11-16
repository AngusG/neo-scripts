import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

#/dev/video0 is vadc composite

video_src = cv2.VideoCapture(1)
#optional
#video_src.set(3, 320)
#video_src.set(4, 240)

while True:

	ret, frame = video_src.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray, 
		scaleFactor = 1.2,
		minNeighbors = 5,
		minSize(30, 30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	#Draw rectangle around faces
	for(x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# Display

	cv2.imshow("video", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_src.release()
cv2.destroyAllWindows()
