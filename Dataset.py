import os
import torch
from PIL import Image
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset

TARGET_SIZE = (700, 400)  # Szerokość x Wysokość

def resize_and_pad(image, target_size=TARGET_SIZE, fill=0, resize_mode=Image.BICUBIC):
    """Resize obrazu z zachowaniem proporcji i paddingiem."""
    w, h = image.size
    ratio = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * ratio), int(h * ratio)

    image = image.resize((new_w, new_h), resize_mode)

    pad_w = (target_size[0] - new_w) // 2
    pad_h = (target_size[1] - new_h) // 2
    padding = (pad_w, pad_h, target_size[0] - new_w - pad_w, target_size[1] - new_h - pad_h)

    return transforms.functional.pad(image, padding, fill=fill)

# Transformacje bez losowych obrotów
image_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

mask_transforms = transforms.Compose([
    transforms.ToTensor(),
])

class FloodDataset(Dataset):
    def __init__(self, base_folder, transform=None, target_size=TARGET_SIZE, cache_dir='cache'):
        self.base_folder = base_folder
        self.target_size = target_size
        self.cache_dir = cache_dir

        self.image_folder = os.path.join(base_folder, 'train_images')
        self.mask_folder = os.path.join(base_folder, 'train_masks')

        self.images = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg')]

        # Sprawdzanie czy istnieje folder do cache i jego tworzenie
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def __len__(self):
        return len(self.images)

    def _get_cache_path(self, img_name):
        return os.path.join(self.cache_dir, img_name.replace('.jpg', '.pt'))

    def __getitem__(self, idx):
        img_name = self.images[idx]
        cache_path = self._get_cache_path(img_name)

        # Sprawdzamy, czy plik już istnieje w cache
        if os.path.exists(cache_path):
            # Jeśli istnieje, wczytujemy go z cache
            data = torch.load(cache_path)
            image, mask = data['image'], data['mask']
        else:
            # W przeciwnym razie przetwarzamy obraz i zapisujemy do cache
            img_path = os.path.join(self.image_folder, img_name)
            mask_path = os.path.join(self.mask_folder, img_name.replace('.jpg', '.png'))

            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            # Resize z różnymi metodami dla obrazu i maski
            image = resize_and_pad(image, self.target_size, fill=(0, 0, 0), resize_mode=Image.BICUBIC)
            mask = resize_and_pad(mask, self.target_size, fill=0, resize_mode=Image.NEAREST)  # NEAREST dla maski

            # Zastosuj te same transformacje geometryczne
            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if random.random() < 0.15:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

            # Pozostałe transformacje
            image = image_transforms(image)
            mask = mask_transforms(mask)

            # Zapisz dane do cache
            torch.save({'image': image, 'mask': mask}, cache_path)

        return image, mask
