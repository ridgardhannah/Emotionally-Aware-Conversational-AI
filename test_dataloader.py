import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class BP4DDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        
        self.au_columns = ["AU01", "AU02", "AU04", "AU06", "AU07", "AU10", 
                           "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        attempts = 0
        while attempts < 10: 
            if idx >= len(self.data_frame):
                idx = 0

            original_path = str(self.data_frame.iloc[idx, 0])

            parts = original_path.replace("\\", "/").split("/")
            
            try:
                subject = parts[-3]
                task = parts[-2]
                filename = parts[-1]
                
                local_path = f"C:\\Users\\n1246442\\Projects\\BP4D_local\\{subject}\\{task}\\{filename}"
                
                image = Image.open(local_path).convert('RGB')
                
                labels = self.data_frame.iloc[idx][self.au_columns].values.astype('float32')
                labels = torch.tensor(labels)

                if self.transform:
                    image = self.transform(image)

                return image, labels

            except Exception as e:
                idx = (idx + 1) % len(self.data_frame)
                attempts += 1
        
        raise RuntimeError(f"Dataloader failed after 10 skips. Last error: {e}")

if __name__ == "__main__":
    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    print("Checking dataloader connection...")
    dataset = BP4DDataset(csv_file="subset_training_data.csv", transform=test_transform)
    img, lbl = dataset[0]
    print(f"done. Image shape: {img.shape}, Labels: {lbl}")