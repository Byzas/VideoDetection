import cv2
import operator



faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture("DiorPortman.mp4")


# the output will be written to resultHaarCascades.avi
out = cv2.VideoWriter(
    'resultHaarCascades.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (720, 576))

while(cap.isOpened()):
    _, frame = cap.read()

    resizedImage = cv2.resize(frame, (720,576))
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)


    #detect the face using the haarcascade_frontalface_alt.xml 
    face = faceCascade.detectMultiScale(grayImage, scaleFactor=1.2, minNeighbors=5)
    for (x, y, width, height) in face:
        cv2.rectangle(resizedImage, (x, y), (x+width, y+height), (0,0, 255), 2)

    # Write the output video
    out.write(resizedImage.astype('uint8'))
    cv2.imshow("res", resizedImage)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
