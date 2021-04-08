import cv2

imagePath="apolo11.jpg"

cascadeClassifierPath = "haarcascade_frontalface_alt.xml"
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)

image = cv2.imread(imagePath)

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detectedFaces = cascadeClassifier.detectMultiScale(grayImage)

for(x,y,width,height) in detectedFaces:
    cv2.rectangle(image, (x, y), (x+width, y+width), (0,0, 255), 5)

cv2.imwrite("res.jpg", image)