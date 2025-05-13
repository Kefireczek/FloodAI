import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import directed_hausdorff

def dice_score(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def hausdorff_distance(pred, target):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    pred_coords = np.argwhere(pred_np > 0)
    target_coords = np.argwhere(target_np > 0)

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return np.nan  # albo duża liczba jako kara

    hd1 = directed_hausdorff(pred_coords, target_coords)[0]
    hd2 = directed_hausdorff(target_coords, pred_coords)[0]
    return max(hd1, hd2)

def plot_losses(train_losses, val_losses, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, num_epochs + 1))
    plt.show()

def plot_metrics(dice_scores, iou_scores, hausdorff_scores, num_epochs):
    epochs = list(range(1, num_epochs + 1))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Oś dla Dice i IoU
    ax1.plot(epochs, dice_scores, label='Dice', marker='o', color='tab:blue')
    ax1.plot(epochs, iou_scores, label='IoU', marker='s', color='tab:green')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Dice / IoU', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(epochs)
    ax1.grid(True)

    # Druga oś dla Hausdorffa
    ax2 = ax1.twinx()
    ax2.plot(epochs, hausdorff_scores, label='Hausdorff', marker='^', color='tab:red')
    ax2.set_ylabel('Hausdorff distance', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Połącz legendy z obu osi
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('Metryki walidacyjne na przestrzeni epok')
    plt.tight_layout()
    plt.show()