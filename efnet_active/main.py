import torch
import cv2
import torchvision
from torchvision import transforms
from pathlib import Path
import time

save_path = Path(__file__).parent


def build_model(model_path):
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)

    features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(features, 1)

    if model_path == None and model_path.exists():
        model.load_state_dict(torch.load(model_path))
    
    return model

model_file = save_path / "model.pth"
model = build_model(model_file)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])])

def predict(frame):
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()
    return "Person" if prob > 0.5 else "No Person", prob

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

while True:
    _, frame = cap.read()
    if frame is None:
        break
    
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    elif key == ord("p"):
        t = time.time()
        label, confidence = predict(frame)
        print(f"Elapsed Time: {time.time() - t:.4f}s, Prediction: {label}, Confidence: {confidence:.2f}")

cap.release()
cv2.destroyAllWindows()