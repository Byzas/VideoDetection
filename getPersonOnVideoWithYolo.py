import cv2
import numpy as np

confidenceVar = 0.5
threshold = 0.3

## Init Yolo DNN
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names.txt", "r") as file:
    classes = [line.strip() for line in file.readlines()]
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

redColors = (0, 0, 255)

# Init video stream
cap = cv2.VideoCapture("DiorPortman.mp4")

# Get the frame width/height to write the new video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# the output will be written to result.avi
out = cv2.VideoWriter(
    'resultYolov3.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (frame_width, frame_height))

while (True):
    _, frame = cap.read()

    height, width = frame.shape[:2]
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes and classes[class_ids[i]] == "person":
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}%".format(classes[class_ids[i]], confidences[i] * 100)
            cv2.rectangle(frame, (x, y), (x + w, y + h), redColors, 3)
            cv2.putText(frame, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 4, redColors, 4)

    # Write the output video
    out.write(frame.astype('uint8'))
    cv2.imshow("res", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
# and release the output
out.release()
cv2.destroyAllWindows()
