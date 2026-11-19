import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import pandas as pd
import os
from tqdm import tqdm
from test_dataloader import BP4DDataset

#settings
BATCH_SIZE = 16 
EPOCHS = 15
LEARNING_RATE = 0.0001
CHECKPOINT_PATH = "best_vision_model_EXPANDED_FINAL.pth"
CSV_FILE = "subset_training_data.csv"

#masked bce function
def masked_bce_loss(outputs, labels):
    mask = (labels != 9).float()
    safe_labels = labels.clone()
    safe_labels[labels == 9] = 0.0
    bce = nn.BCEWithLogitsLoss(reduction='none')
    loss = bce(outputs, safe_labels)
    masked_loss = loss * mask
    return masked_loss.sum() / (mask.sum() + 1e-8)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"starting training on {device}")

    #stochastic augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BP4DDataset(csv_file=CSV_FILE, transform=train_transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(dataset.au_columns))
    )
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')
    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = masked_bce_loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")

        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        
        history.append({
            "Epoch": epoch + 1,
            "Average_Loss": epoch_loss,
            "New_Best_Saved": is_best
        })
        pd.DataFrame(history).to_csv("resnet18_training_curve.csv", index=False)

    print("Training done")

if __name__ == "__main__":
    train_model()