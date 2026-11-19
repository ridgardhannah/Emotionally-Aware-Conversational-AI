import torch
import torch.nn as nn
from torchvision import models

def test_architecture():
    print("Testing ResNet-18 Dimensionality")
    
    #model architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 12) #12 Action Unit targets
    )
    model.eval()

    #create a fake dummy image (1 batch, 3 RGB channels, 224x224 pixels)
    dummy_image = torch.randn(1, 3, 224, 224)
    print(f"Input Tensor Shape:  {list(dummy_image.shape)} -> [Batch, Channels, Height, Width]")

    #pass dummy image through the network
    with torch.no_grad():
        output = model(dummy_image)
    
    print(f"Output Tensor Shape: {list(output.shape)} -> [Batch, Action_Units]")

    #verificatin check
    print("\n Verification")
    if output.shape == (1, 12):
        print("SUCCESS: Architecture outputs a 12-dimensional vector.")
        print("Verified compliance with the AU target space defined in Section 4.3.5.")
    else:
        print("ERROR: Output shape mismatch")

if __name__ == "__main__":
    test_architecture()