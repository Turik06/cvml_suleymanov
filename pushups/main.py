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


def detect_push_up(annotated, keypoints, is_down, count):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    trigger = False

    if left_shoulder[0] > 0 and right_shoulder[0] > 0:
        l_angle = get_angle(left_shoulder, left_elbow, left_wrist)
        r_angle = get_angle(right_shoulder, right_elbow, right_wrist)

        if l_angle < 100 and r_angle < 100 and not is_down:
            is_down = True
        
        if l_angle > 150 and r_angle > 150 and is_down:
            is_down = False
            count += 1
            trigger = True
            
        cv2.putText(annotated, f"Pushups: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    return is_down, count, trigger

model = YOLO("yolo26n-pose.pt")
model.to("cuda")

camera = cv2.VideoCapture(0)
ps = None
sound_file = r"D:\education\CV\HW\pushups\acolyteyes2.mp3" 

is_down = False
count = 0
last_seen_time = time.time()

while camera.isOpened():
    ret, frame = camera.read()
    if time.time() - last_seen_time > 5.0:
        count = 0
        is_down = False
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

    t = time.perf_counter()
    results = model.predict(frame)
    
    
    print(f"Elapsed time {1 / (time.perf_counter() - t):.1f}")

    if not results:
        continue
    result = results[0]
    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue
        
    if len(keypoints[0]) < 11:
        continue

    print(keypoints)
    last_seen_time = time.time()
    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()
    is_down, count, trigger = detect_push_up(annotated, keypoints[0], is_down, count)
    
    if trigger:
        if ps is None:
            ps = playsound(sound_file, block=False)
        else:
            if not ps.is_alive():
                ps = playsound(sound_file, block=False)

    cv2.imshow("Pose", annotated)

camera.release()
cv2.destroyAllWindows()