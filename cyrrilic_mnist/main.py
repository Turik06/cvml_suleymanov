import torch, random, zipfile
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from train_model import CyrillicCNN, CyrrilicDataset, get_zip_info

root = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
zip_path, model_path = root / 'cyrillic.zip', root / "model.pth"

all_paths, class_to_idx = get_zip_info(zip_path)
idx_to_class = {v: k for k, v in class_to_idx.items()}
all_labels = [class_to_idx[p.split('/')[1]] for p in all_paths]

_, test_paths = train_test_split(all_paths, test_size=0.2, random_state=42, stratify=all_labels)

test_loader = DataLoader(CyrrilicDataset(zip_path, test_paths, class_to_idx, transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((28, 28)), 
    transforms.Normalize((0.5,), (0.5,))
])), batch_size=64, shuffle=True)

model = CyrillicCNN().to(device)
if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

correct, total = 0, 0
plt.figure(figsize=(10, 10))

with torch.no_grad():
    for b_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        
        if b_idx == 0:
            for i in range(min(16, len(images))):
                plt.subplot(4, 4, i + 1)
                p_c, r_c = idx_to_class[preds[i].item()], idx_to_class[labels[i].item()]
                plt.title(f"Predict: {p_c} | Real: {r_c}")
        
        total += labels.size(0)
        correct += (preds == labels).sum().item()

with torch.no_grad():
    for b_idx, (images, labels) in enumerate(test_loader):
        preds = model(images).argmax(1)
        
        if b_idx == 0:
            for i in range(min(16, len(images))):
                plt.subplot(4, 4, i + 1)
                img = images[i].squeeze() * 0.5 + 0.5
                plt.imshow(img, cmap='gray')
                
                p_c, r_c = idx_to_class[preds[i].item()], idx_to_class[labels[i].item()]
                plt.title(f"Predict: {p_c} | Real: {r_c}")
        
        total += labels.size(0)
        correct += (preds == labels).sum().item()

plt.savefig(root / "prediction.png")
print(f"Accuracy: {100 * correct / total}%")
plt.show()