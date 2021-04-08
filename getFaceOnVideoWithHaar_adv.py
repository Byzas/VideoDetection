import cv2
import operator



faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
profileCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
upperBodyCascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
fullBodyCascase = cv2.CascadeClassifier("haarcascade_fullbody_alt.xml")

cap = cv2.VideoCapture("DiorPortman.mp4")

widthBased = 576

neighborMargin = 140


# the output will be written to resultHaarCascades.avi
out = cv2.VideoWriter(
    'resultHaarCascades.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (720, 576))

while(cap.isOpened()):
    _, frame = cap.read()

    detectList = []

    resizedImage = cv2.resize(frame, (720,576))
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)


    # Detect firstly a full body person, a bust of person and finally the face
    detectPerson = fullBodyCascase.detectMultiScale(grayImage, scaleFactor = 1.2, minNeighbors= 5)
    for(x,y,width,height) in detectPerson:
        detectList.append([x, y, x+width, y+height])

    if detectList is None:
        detectPerson = upperBodyCascade.detectMultiScale(grayImage, scaleFactor=1.2, minNeighbors=5)
        for (x, y, width, height) in detectPerson:
            detectList.append([x, y, x+width, y+height])

    if detectList is None:
        detectPerson = faceCascade.detectMultiScale(grayImage, scaleFactor=1.2, minNeighbors=5)
        for (x, y, width, height) in detectPerson:
            detectList.append([x, y, x+width, y+height])

        detectPerson = profileCascade.detectMultiScale(grayImage, scaleFactor=1.2, minNeighbors=5)
        for (x, y, width, height) in detectPerson:
            detectList.append([x, y, x+width, y+height])

        grayImage2 = cv2.flip(grayImage, 1)
        detectPerson = profileCascade.detectMultiScale(grayImage2, scaleFactor=1.2, minNeighbors=5)
        for (x, y, width, height) in detectPerson:
            detectList.append([widthBased-x, y, widthBased-x+width, y+height])

    detectList = sorted(detectList, key=operator.itemgetter(0,1))
    index = 0
    for x, y, x2, y2 in detectList:
        if not index or (x-detectList[index-1][0]>neighborMargin or y-detectList[index-1][1]>neighborMargin):
            cv2.rectangle(resizedImage, (x, y), (x2, y2), (0, 0, 255), 2)
        index += 1

    # Write the output video
    out.write(resizedImage.astype('uint8'))
    cv2.imshow("res", resizedImage)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
