import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import FloodDataset
from Model import FloodUNet
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

print(torch.cuda.is_available())  # powinno być True
print(torch.cuda.get_device_name(0))  # powinna być twoja karta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_positive_pixel_ratio(loader):
    total_pixels = 0
    total_positive = 0

    for _, masks in loader:
        masks = masks.view(-1)  # spłaszczamy wszystko
        total_pixels += masks.numel()
        total_positive += masks.sum().item()  # liczba "1" w maskach

    ratio = total_positive / total_pixels
    return ratio


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # bo model zwraca logits
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def combined_loss(pred, target, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).float()  # maski binarne: 0 lub 1

        outputs = model(images)
        loss = combined_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def visualize_sample(model, loader):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(loader))
        images = images.to(device)
        outputs = torch.sigmoid(model(images)).cpu().numpy()
        outputs = (outputs > 0.5).astype(np.uint8)

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        for i in range(3):
            axs[i, 0].imshow(images[i].permute(1, 2, 0).cpu())
            axs[i, 1].imshow(masks[i].squeeze(), cmap='gray')
            axs[i, 2].imshow(outputs[i].squeeze(), cmap='gray')
            axs[i, 0].set_title("Obraz")
            axs[i, 1].set_title("Maska GT")
            axs[i, 2].set_title("Predykcja")
            for j in range(3): axs[i, j].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    dataset = FloodDataset(base_folder='train_data', transform=None)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    ratio=compute_positive_pixel_ratio(loader)
    print("ratio = ",ratio)

    model = FloodUNet().to(device)
    print("===> Architektura modelu U-Net:")
    summary(model, input_size=(1, 3, 400, 700))  # (batch_size, channels, height, width)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5 # Zmieniasz tutaj liczbę epok

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f}")

    # Wizualizacja predykcji
    visualize_sample(model, loader)