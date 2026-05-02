import torch
import matplotlib.pyplot as plt
from train import Encoder, Decoder, ImageDataset

def test_model(mode, device):
    print(f"Тестируем режим {mode}")
  
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    encoder.load_state_dict(torch.load(f"encoder_{mode}.pth", map_location=device))
    decoder.load_state_dict(torch.load(f"decoder_{mode}.pth", map_location=device))
    
    encoder.eval()
    decoder.eval()
    
    dataset = ImageDataset(n=1, size=256, mode=mode)
    original_img, _ = dataset[0]
    
    with torch.no_grad():
        img_tensor = original_img.unsqueeze(0).to(device)
        latent = encoder(img_tensor)
        reconstructed = decoder(latent)
        
    orig_show = original_img[0].cpu().numpy()
    recon_show = reconstructed[0][0].cpu().numpy()
    
    return orig_show, recon_show

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    fig.suptitle("Оригинал vs Восстановление", fontsize=16)
    
    for i, mode in enumerate([1, 2, 3, 4]):
        orig, recon = test_model(mode, device)
        
        axes[i, 0].imshow(orig, cmap='gray')
        axes[i, 0].set_title(f"Режим {mode} (Оригинал)")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(recon, cmap='gray')
        axes[i, 1].set_title(f"Режим {mode} (Восстановлено)")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig("results_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()