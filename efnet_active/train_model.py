import torch
import cv2
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from collections import deque   
from pathlib import Path


save_path = Path(__file__).parent
def build_model(model_path):
    weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    model = torchvision.models.alexnet(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(features, 1)

    if model_path == None and model_path.exists():
        model.load_state_dict(torch.load(model_path))
    
    return model


model_file = save_path / "model.pth"
model = build_model(model_file)
print(model)



criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    model.parameters()), 
                                    lr=0.001)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])])


def train(buffer):
    if len(buffer) < 10:
        return None
    model.train()
    images, labels = buffer.get_batch()
    optimizer.zero_grad()
    predictions = model(images).squeeze()
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(frame):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, 
                                    cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()
    label = "Person" if prob > 0.5 else "No Person"
    return label, prob

class Buffer:
    def __init__(self, maxsize = 16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)

    def append(self,frame,label):
        self.frames.append(frame)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)
    
    def get_batch(self):
        images = torch.stack(list(self.frames))
        labels = torch.tensor(list(self.labels), dtype=torch.float32)
        return images, labels
    

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera",cv2.WINDOW_GUI_NORMAL)
buffer = Buffer()
count_labeled = 0

while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if key == ord("q"):
        break
    elif key == ord("1"): # person
        tensor = transform(image)
        buffer.append(tensor, 1.0)
        count_labeled += 1
    elif key == ord("2"): # no person
        tensor = transform(image)
        buffer.append(tensor, 0.0)
        count_labeled += 1
    elif key == ord("p"): # predict
        t= time.time()
        label ,confidence = predict(frame)
        print(f"Elapsed Time: {time.time() - t} seconds, Prediction: {label}, Confidence: {confidence}")
    elif key == ord("s"): # save model
        torch.save(model.state_dict(),save_path / "model.pth") 

    print(f"Buffer Size: {len(buffer),count_labeled}")
    if count_labeled >= buffer.frames.maxlen:
        loss = train(buffer)
        if loss:
            print(f"Loss = {loss}")
        count_labeled = 0
