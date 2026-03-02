import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread
from collections import Counter

def extractor(image):
  if image.ndim == 2:
    binary = image
  else:
    gray = np.mean(image, 2).astype("u1")
    threshold = 4
    binary = gray > threshold
  lb = label(binary)
  props = regionprops(lb)

  h,w = props[0].image.shape
  cy,cx = props[0].centroid_local
  cy /= h
  cx /= w
  return np.array([props[0].eccentricity,props[0].euler_number,props[0].extent,cy,cx], dtype = 'f4')

def make_train(path):
  train = []
  responses = []
  ncls = 0
  word = {}
  for cls in sorted(path.glob("*")):
#     print(cls)
    ncls += 1
    name = cls.name
    if name.startswith('s') and len(name) > 1:
        name = name[1:]

    word[ncls] = name
    print(cls.name,ncls)
    for p in cls.glob("*.png"):
#       print(p)
      train.append(extractor(imread(p)))
      responses.append(ncls)
  train = np.array(train, dtype = "f4").reshape(-1, 5)
  responses = np.array(responses, dtype = 'f4').reshape(-1, 1)
  return train, responses, word

data = Path("D:\education\CV\HW")

train,responses, letters = make_train(data / "task/train")
knn = cv2.ml.KNearest.create()
knn.train(train, cv2.ml.ROW_SAMPLE, responses)

w = []
result = []
for i in range(7):
    img = data / "task" / f"{i}.png"
    
    img = imread(img)
    gray = img.mean(2)
    binary = gray > 4 


    lb = label(binary)
    
    props = sorted(regionprops(lb), key=lambda x: x.bbox[1])
    
    merged_boxes = []
    eps = 10
    
    for prop in props:
        y1, x1, y2, x2 = prop.bbox
        cy, cx = prop.centroid
        
        if merged_boxes:
            last_y1, last_x1, last_y2, last_x2, last_cx = merged_boxes[-1]
            if abs(cx - last_cx) < eps:
                merged_boxes[-1] = (
                    min(y1, last_y1), min(x1, last_x1), 
                    max(y2, last_y2), max(x2, last_x2),
                    last_cx
                )
                continue
                
        merged_boxes.append((y1, x1, y2, x2, cx))
        
    res = ""
    x2_p = None
    threshold_bbox = 20
    for bbox in merged_boxes:        
        y1, x1, y2, x2, _ = bbox

        if x2_p is not None:
           distance = x1 - x2_p
           if distance > threshold_bbox:
               res += " "
        
        char_image = binary[y1:y2, x1:x2]

        ext = extractor(char_image).reshape(1, 5)
        ret, results, neighbours, dist = knn.findNearest(ext, 5)
        idx = int(results[0][0])
        char = letters[idx]
        res += char
        x2_p = x2
        
    print(res)
    result.append(res)

phrases_true = [
  'C is LOW-LEVEL',
  'C++ is POWERFUL',
  'Python is INTUITIVE',
  'Rust is SAFE',
  'LUA is EASY',
  'Javascript is UGLY',
  'PHP sucks'
]

char_matches = 0
total_chars = sum(len(t) for t in phrases_true)

for pred, true in zip(result, phrases_true):
    char_matches += sum(1 for a, b in zip(pred, true) if a == b)

char_acc = char_matches / total_chars if total_chars > 0 else 0.0
print(f'accuracy {char_acc}')

