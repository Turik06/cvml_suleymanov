import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound
import numpy as np


def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0]) 
    ab = np.atan2(a[1] - b[1], a[0] - b[0]) 
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle

def detect_hands_up(annotated, keypoints):
    nose_seen = (keypoints[0][0] > 0 and 
    keypoints[0][1] > 0 )
    
    eyes_seen = (keypoints[1][0] > 0 and 
    keypoints[1][1] > 0 and
    keypoints[2][0] > 0 and
    keypoints[2][1] > 0 )

    left_shoulder = keypoints[5]
    ritgh_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    ritgh_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    if nose_seen and eyes_seen:
        if ((left_shoulder[1] >
            left_elbow[1] >
            left_wrist[1]) or
            (ritgh_shoulder[1] >
            ritgh_elbow[1] >
            right_wrist[1])):  
            left_angle = get_angle(left_shoulder, left_elbow, left_wrist)   
            cv2.putText(annotated,
                        f"Hands up{left_angle:1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 1)
            return True
    return None

model = YOLO("yolo26n-pose.pt")
model.to("cuda")


camera = cv2.VideoCapture(0)
ps = None
sound_file = r"D:\education\CV\L12\BULLET.mp3"

while camera.isOpened():
    ret, frame = camera.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

    t = time.perf_counter()
    results = model.predict(frame)
    print(f"Elapsed time {1 / (time.perf_counter() - t)}:.1f")

    if not results:
        continue
    result = results[0]
    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue

    print(keypoints)

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape,5,True)
    annotated = annotator.result()
    detect_hands_up(annotated, keypoints[0])
    
    if detect_hands_up(annotated, keypoints[0]):
        if ps is None:
            ps = playsound(sound_file,block=False)
        else:
            if not ps.is_alive():
                ps = None
    cv2.imshow("Pose", annotated)
