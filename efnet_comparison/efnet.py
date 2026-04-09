from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

foldr = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.255]),
])
val_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.255]),
])

train_ds = ImageFolder(foldr / "dataset" / "dataset" / "train", transform=train_transform)
val_ds = ImageFolder(foldr / "dataset" / "dataset" / "val", transform=val_transform)
train_loader = DataLoader(train_ds, 32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, 32, shuffle=False, num_workers=4)

def build_model(model_name):
    if model_name == "b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b0(weights=weights)
    elif model_name == "b1":
        weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b1(weights=weights)
    elif model_name == "b2":
        weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b2(weights=weights)
        
    for p in model.parameters():
        p.requires_grad = False
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 3)
    )
    return model.to(device)

if __name__ == "__main__":
    models_to_test = ["b0", "b1", "b2"]
    results_acc = {}
    epochs = 20
    

    for m_name in models_to_test:
        print(f"Обучение EfficientNet-{m_name}")
        model = build_model(m_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        final_preds, final_labels = [], []

        for epoch in range(epochs):
            model.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss_sum += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += images.size(0)
                
            train_loss = train_loss_sum / train_total
            train_acc = train_correct / train_total

            model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            val_preds, val_labels = [], []
            
            with torch.no_grad():  
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    
                    val_loss_sum += loss.item() * images.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += images.size(0)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    
            val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total

            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} Loss_train: {train_loss} Acc_train: {train_acc}  Loss_val: {val_loss} Acc_val: {val_acc}")
            
            if epoch == epochs - 1:
                final_preds = val_preds
                final_labels = val_labels
                results_acc[m_name] = val_acc
        
        cm = confusion_matrix(final_labels, final_preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=train_ds.classes, yticklabels=train_ds.classes)
        
        plt.xlabel('Предсказанные классы')
        plt.ylabel('Истинные классы')
        plt.title(f'EfficientNet {m_name}')
        plt.tight_layout()
        
        plt.savefig(foldr / f"cm_{m_name}.png", dpi=300)
        plt.close()

    print(f"ИТОГ:")
    for m, acc in results_acc.items():
        print(f"EfficientNet-{m}: {acc * 100}%")