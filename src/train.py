import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import SiameseUNet

class ChangeDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.before_dir = os.path.join(data_dir, "before")
        self.after_dir = os.path.join(data_dir, "after")
        self.mask_dir = os.path.join(data_dir, "masks")
        
        self.image_files = sorted(os.listdir(self.before_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        before_img = Image.open(os.path.join(self.before_dir, filename)).convert("RGB")
        after_img = Image.open(os.path.join(self.after_dir, filename)).convert("RGB")
        
        # Masks are saved as .png to preserve binary pixels
        mask_filename = filename.replace(".jpg", ".png")
        mask_img = Image.open(os.path.join(self.mask_dir, mask_filename)).convert("L")
        
        if self.transform:
            before_img = self.transform(before_img)
            after_img = self.transform(after_img)
            mask_img = self.transform(mask_img)
            
        # Binarize mask
        mask_img = (mask_img > 0).float()
            
        return before_img, after_img, mask_img

def train(data_dir, epochs=50, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = ChangeDetectionDataset(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except FileNotFoundError:
        print(f"Dataset directory '{data_dir}' not found. Please create before, after, and masks folders.")
        return
        
    model = SiameseUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for b_img, a_img, mask in dataloader:
            b_img, a_img, mask = b_img.to(device), a_img.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(b_img, a_img)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "best_model.pth")
    print("Training complete. Saved to best_model.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()
    
    train(args.data_dir, args.epochs)
