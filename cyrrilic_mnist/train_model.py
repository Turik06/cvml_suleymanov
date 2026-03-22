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

class CyrrilicDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.transform = transform
        self.zip_file = zipfile.ZipFile(zip_path, 'r')
        self.image_paths = [f for f in self.zip_file.namelist() if f.endswith('.png')]
        classes = sorted(set(f.split('/')[1] for f in self.image_paths))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2) 
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 3 * 3, 256) 
        self.relu_fc = nn.ReLU()
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
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
save_path = Path(__file__).parent
device = torch.device("cuda" \
                      if torch.cuda.is_available() else "cpu")
print(f"{device=}")
torch.manual_seed(42)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28, 28)),
    transforms.RandomAffine(
        degrees=10, 
        translate=(0.05, 0.05), 
        scale=(0.9, 1.1)
    ), 
    transforms.Normalize((0.5,), (0.5,))
])

zip_file_path = save_path / 'cyrillic.zip'

full_dataset = CyrrilicDataset(zip_path=zip_file_path, transform=train_transform)
num_classes = len(full_dataset.class_to_idx)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

batch_size = 64

train_loader = DataLoader(train_dataset,
                        batch_size=batch_size, 
                        shuffle=True)


test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

model = CyrillicCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)


num_epochs = 20
train_loss = []
train_acc = []

model_path = save_path / "model.pth"

if not model_path.exists():
    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        model.train()
        run_loss = 0.0
        total = 0
        correct = 0

        for batch_idx,(images , labels) in enumerate(train_loader):
            images, labels = (images.to(device),
                              labels.to(device))
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
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        t = time.perf_counter() - start_time
        print(f"Epoch: {epoch+1}, Loss: {epoch_loss}, Acc: {epoch_acc}, Time: {t:.2f} s")

    torch.save(model.state_dict(), model_path)
    plt.figure()
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(train_loss)
    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(train_acc)
    plt.savefig(save_path / "train.png")
    plt.show()
else:
    model.load_state_dict(torch.load(model_path))


model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test sample: {100 * correct / total}")
