from ultralytics import YOLO
from pathlib import Path
import yaml
import torch

classes = {0: "cube", 1: "neither", 2: "sphere"}
root = Path(__file__).parent / "tural-photo" / "ds"

config = {
    "path" : str(root.absolute()),
    "train": str((root/"images/train").absolute()),
    "val": str((root/"images/train").absolute()),
    "nc": len(classes),
    "names": classes  
}

with open(root/"dataset.yaml","w") as f:
    yaml.dump(config,f,allow_unicode = True )

if __name__ == "__main__":
    size = "n" 
    model = YOLO("hw.pt")

    result = model.train(
        data=str(root / "dataset.yaml"),
        imgsz = 640,
        batch = 640,
        workers = 4 ,
        epochs = 10,
        patience = 5,
        optimizer = "AdamW",
        lr0 = 0.01,
        warmup_epochs = 3,
        cos_lr = True 

        dropout = 0.2 
        
        hsv_h = 0.015

        

    )