#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import numpy as np
import torch.optim  as optim
from pathlib import Path
import cv2
import zipfile
from torch.utils.data import DataLoader, Dataset, random_split
import time
from sklearn.model_selection import train_test_split



def get_zip_info(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_paths = [f for f in z.namelist() if f.endswith('.png') and '/' in f]
        classes = sorted(list(set(f.split('/')[1] for f in all_paths)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        return all_paths, class_to_idx
    

class CyrrilicDataset(Dataset):
    def __init__(self, zip_path, image_paths, class_to_idx, transform=None):
        self.transform = transform
        self.zip_path = zip_path
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.zip_file = zipfile.ZipFile(zip_path, 'r')

    def __len__(self):
        return len(self.image_paths)
      
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        img_bytes = np.frombuffer(self.zip_file.read(path), np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
        
        image = image[:, :, 3]
        
        image = np.expand_dims(image, axis=-1)
        label = self.class_to_idx[path.split('/')[1]]
        
        if self.transform:
            image = self.transform(image)
        return image, label
    

class CyrillicCNN(nn.Module):

    def __init__(self):
        super(CyrillicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2) 
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256) 
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 34)

    def forward(self, x):
       
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
 
if __name__ == "__main__":
    save_path = Path(__file__).parent
    zip_file_path = save_path / 'cyrillic.zip'
    model_path = save_path / "model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")


    path_all, class_to_idx = get_zip_info(zip_file_path) 
    all_labels = [class_to_idx[f.split('/')[1]] for f in path_all]

    train_paths, test_paths, train_labels, _ = train_test_split(
        path_all, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)), 
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CyrrilicDataset(zip_file_path, train_paths, class_to_idx, train_transform)
    test_dataset = CyrrilicDataset(zip_file_path, test_paths, class_to_idx, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CyrillicCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

    train_loss_hist = []
    train_acc_hist = []

    if not model_path.exists():
        for epoch in range(20):
            start_time = time.perf_counter()
            model.train()
            run_loss, correct, total = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                run_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            scheduler.step()
            epoch_loss = run_loss / len(train_loader)
            epoch_acc = 100 * (correct / total)
            train_loss_hist.append(epoch_loss)
            train_acc_hist.append(epoch_acc)
            
            t = time.perf_counter() - start_time
            print(f"Epoch: {epoch+1}, Loss: {epoch_loss}, Acc: {epoch_acc}%, Time: {t} s")

        torch.save(model.state_dict(), model_path)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121); plt.title("Loss"); plt.plot(train_loss_hist)
        plt.subplot(122); plt.title("Accuracy"); plt.plot(train_acc_hist)
        plt.savefig(save_path / "train.png")
        plt.show()
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")
    