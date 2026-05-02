import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as transorm
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import numpy as np
import time

class ImageDataset(Dataset):
    def __init__(self,n=200,size = 128):
        super().__init__()
        self.n = n
        self.size = size
        self.transform = transorm.Compose([
            transorm.ToTensor(),
            transorm.Normalize(mean=[0.5],std=[0.5])
        ])

    def __len__(self):
        return self.n
    
    def __getitem__(self,idx):
        image = Image.new("L",
                          (self.size,self.size),
                          color = 255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = "ABC"
        # x = np.random.randint(10,self.size-40)
        # y = np.random.randint(10,self.size-40)
        x = 30
        y = 30
        draw.text((x,y),text,fill=0,font=font)

        tensor = self.transform(image)
        return tensor,tensor
ds = ImageDataset()
plt.imshow(ds[0][0][0])
plt.show()


class Encoder(nn.Module):
    def __init__(self,latent = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size= 4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.bottleneck = nn.Linear(256 * 16 * 16,latent)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.bottleneck(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,latent = 512):
        super().__init__()
        self.bottleneck = nn.Linear(latent,256 * 16 * 16)  

        self.features = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32,1,kernel_size=4,stride=2,padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.bottleneck(x)
        x = x.view(x.size(0),256,16,16)
        x = self.features(x)
        return x
    
encoder = Encoder()
decoder = Decoder()
# print(encoder)
# print(decoder)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Параметров в Encoder: {count_parameters(encoder):,}")
print(f"Параметров в Decoder: {count_parameters(decoder):,}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    
    dataset = ImageDataset(20000, 256)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

    encoder.train()
    decoder.train()

    epochs = 10
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            latent = encoder(imgs)
            output = decoder(latent)
            
            loss = criterion(output, imgs) 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Time: {time.time()-epoch_start:.2f}s")

    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")
