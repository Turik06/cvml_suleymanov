import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from train_model import CyrillicCNN, CyrrilicDataset

save_path = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CyrrilicDataset(
    zip_path=save_path / 'cyrillic.zip', 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))
    ])
)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}


model = CyrillicCNN().to(device)
model.load_state_dict(torch.load(save_path / "model.pth", map_location=device))
model.eval() 

plt.figure(figsize=(15, 5))
with torch.no_grad():
    for i, idx in enumerate(random.sample(range(len(dataset)), 5)):
        image, label = dataset[idx]
        
        pred_idx = model(image.unsqueeze(0).to(device)).argmax(1).item()

        real_char = idx_to_class[label]
        pred_char = idx_to_class[pred_idx]

        plt.subplot(1, 5, i + 1)
        plt.imshow(image.squeeze() * 0.5 + 0.5)
        plt.title(f"Predict:{pred_char} | Real:{real_char}")

plt.tight_layout()
plt.show()