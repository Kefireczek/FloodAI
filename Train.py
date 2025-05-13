import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import FloodDataset, TARGET_HEIGHT, TARGET_WIDTH
from Metrics import plot_losses, dice_score, iou_score, hausdorff_distance, plot_metrics
from Model import FloodUNet
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

def train_one_epoch(model, loader, optimizer):
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


def evaluate_one_epoch(model, loader):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_hausdorff = 0
    num_samples = 0
    num_hausdorff_samples = 0

    hausdorff_every_n = 2

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).float()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5

            for i, (pred, target) in enumerate(zip(preds, masks)):
                total_dice += dice_score(pred, target).item()
                total_iou += iou_score(pred, target).item()
                num_samples += 1

                if num_samples % hausdorff_every_n == 0:
                    hd = hausdorff_distance(pred, target)
                    if not np.isnan(hd):
                        total_hausdorff += hd
                        num_hausdorff_samples += 1

    return {
        "loss": total_loss / len(loader),
        "dice": total_dice / num_samples if num_samples > 0 else 0,
        "iou": total_iou / num_samples if num_samples > 0 else 0,
        "hausdorff": total_hausdorff / num_hausdorff_samples if num_hausdorff_samples > 0 else float('nan')
    }

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
    train_dataset = FloodDataset(base_folder='train_data', transform=None)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    val_dataset = FloodDataset(base_folder='valid_data', transform=None)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = FloodUNet().to(device)
    print("===> Architektura modelu U-Net:")
    summary(model, input_size=(1, 3, TARGET_HEIGHT, TARGET_WIDTH))  # (batch_size, channels, height, width)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Scheduler zmniejszy LR o połowę, jeśli val_loss nie poprawi się przez 3 epoki
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)


    num_epochs = 10 # Zmieniasz tutaj liczbę epok

    train_losses = []
    val_losses = []
    dice_scores = []
    iou_scores = []
    hausdorff_scores = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = evaluate_one_epoch(model, val_loader)
        val_metrics = evaluate_one_epoch(model, val_loader)

        # Zmniejszenie LR jeśli val_loss się nie poprawia
        scheduler.step(val_metrics['loss'])

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        dice_scores.append(val_metrics['dice'])
        iou_scores.append(val_metrics['iou'])
        hausdorff_scores.append(val_metrics['hausdorff'])

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_metrics['loss']:.4f} - "
              f"Dice: {val_metrics['dice']:.4f} - "
              f"IoU: {val_metrics['iou']:.4f} - "
              f"Hausdorff: {val_metrics['hausdorff']:.2f}")


    # Wizualizacja predykcji
    visualize_sample(model, val_loader) # predykcja ze zbioru walidacyjnego
    plot_losses(train_losses, val_losses,num_epochs)
    plot_metrics(dice_scores, iou_scores, hausdorff_scores, num_epochs)