
from facial_emotion_recognition import EmotionRecognition
import cv2

er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    success, frame = cam.read()

    if not success:
        print("Error: Cannot read webcam frame")
        break

    frame = er.recognise_emotion(frame, return_type='BGR')
    cv2.imshow("Face Emotion Recognition", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press Esc to exit
        break

cam.release()
cv2.destroyAllWindows()

