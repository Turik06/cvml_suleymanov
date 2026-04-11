import cv2
import time
from ultralytics import YOLO
from pathlib import Path

root = Path(__file__).parent / "runs" / "detect" / "figures" / "yolo" / "weights" / "best.pt"
model = YOLO(str(root))
cap = cv2.VideoCapture(0)
cv2.namedWindow("YOLO",cv2.WINDOW_GUI_NORMAL)
predict_mode = True

while True:

    _, frame = cap.read()
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    elif key == ord("p"):
        predict_mode = not predict_mode
        print(f"Predict: {predict_mode}")

    if predict_mode:
        t = time.time()
        results = model(frame, conf=0.2, verbose=False)
        elapsed = time.time() - t
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                print(f"Time: {elapsed:.3f}s | Obj: {name} | Conf: {conf:.2f}")

    cv2.imshow("YOLO", frame)

cap.release()
cv2.destroyAllWindows()