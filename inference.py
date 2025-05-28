import torch
from Model import FloodUNet
from Dataset import TARGET_SIZE, resize_and_pad
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Załaduj model
model = FloodUNet().to(device)
model.load_state_dict(torch.load('flood_unet.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_resized = resize_and_pad(image, target_size=TARGET_SIZE, fill=(0, 0, 0))

    input_tensor = transform(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

    return image_resized, pred_mask


if __name__ == "__main__":
    image_path = "test_img2.png"
    original, mask = predict_image(image_path)

    # Maska jako kolor (np. czerwony) z przezroczystością
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_rgba[mask == 1] = [255, 0, 0, 100]  # czerwony z alpha=100/255

    # Obraz RGB -> RGBA
    original_rgba = original.convert("RGBA")
    overlay = Image.alpha_composite(original_rgba, Image.fromarray(mask_rgba))

    # Wizualizacja
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original)
    axs[0].set_title("Obraz (dopasowany)")

    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Predykcja maski")

    axs[2].imshow(overlay)
    axs[2].set_title("Maska nałożona na obraz")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
