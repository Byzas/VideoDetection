# VideoDetection
Video detection using opencv on Python


Hello visitor,

In this repository you will find some exemple of Video detection using OpenCV.


To install openCV on your python library : pip install opencv-python


In this repository you have several script and files:


apolo11.jpg             		Image test of face detection

coco.names				list of detectable object by the Yolo Dnn

DiorPortman.mp4				Video test of face/person detection

getFaceOnImage.py			Detect the visage in the associated image, raw standard image is apolo11.jpg. It use the Viola-jones algorithm  using haarcascade file.

getFaceOnVideoWithHaar_basic.py		Detect the visage in the associated video, frame by frame, raw standard video is DiorPortman.mp4 . It use the Viola-Jones algorithm using haarcascade file "frontal face".

getFaceOnVideoWithHaar_adv.py		Detect the visage in the associated video, frame by frame, raw standard video is DiosPortman.mp4, It use the viola-jones algorithm useing haarcascade file "front face","full_body","profile_face" and the "upperbody"

getFaceWithHOG.py			Detect full person (person who is fully visible). using HOG algorithm.

getPersonOnVideoWithYolo		Detect all person, using a deep neuronnal network "You Only Look Once", it was adapted to display only person, it take a bit more time by frame but it's more efficient than Viola-Jones algorithm.

haarcascade_frontalface_alt.xml
haarcascade_fullbody_alt.xml
haarcascade_profileface.xml
haarcascase_upperbody.xml		Xml used to apply Viola-Jones algorithm to determine the haar feature used to identify part of visage/person. It's used by getFaceOnVideoWithHaar_basic.py and getFaceOnVideoWithHaar_adv.py

yolov3.cfg				config file use by opencv to apply Yolo algorithm

yolov3.weights				The deep neuronnal network used to apply Yolo Algorithm with the differents weights for each neuronne used.




How to launch.

Just launch the python script with the linked config files, it doesn't need any parameters.

