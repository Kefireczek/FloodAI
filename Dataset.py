import os

import numpy as np
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

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.images = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg')]
        self.cache_files = []


        for img_name in self.images:
            base_name = img_name.replace('.jpg', '')

            # Ścieżki
            original_cache = os.path.join(self.cache_dir, f'{base_name}_orig.pt')
            augment_cache = os.path.join(self.cache_dir, f'{base_name}_aug.pt')

            # Wczytaj tylko jeśli jeszcze nie ma plików w cache
            if not os.path.exists(original_cache) or not os.path.exists(augment_cache):
                img_path = os.path.join(self.image_folder, img_name)
                mask_path = os.path.join(self.mask_folder, base_name + '.png')

                image = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')
                mask = mask.point(lambda p: 255 if p > 128 else 0)
                print(np.unique(np.array(mask)))

                # Resize bez augmentacji
                img_resized = resize_and_pad(image, self.target_size, fill=(0, 0, 0), resize_mode=Image.BICUBIC)
                mask_resized = resize_and_pad(mask, self.target_size, fill=0, resize_mode=Image.NEAREST)

                torch.save({
                    'image': transforms.ToTensor()(img_resized),
                    'mask': transforms.ToTensor()(mask_resized)
                }, original_cache)

                # Z augmentacją (geometria + jittery)
                aug_img = img_resized
                aug_mask = mask_resized

                if random.random() < 0.5:
                    aug_img = transforms.functional.hflip(aug_img)
                    aug_mask = transforms.functional.hflip(aug_mask)
                if random.random() < 0.15:
                    aug_img = transforms.functional.vflip(aug_img)
                    aug_mask = transforms.functional.vflip(aug_mask)

                aug_img = image_transforms(aug_img)
                aug_mask = mask_transforms(aug_mask)

                torch.save({'image': aug_img, 'mask': aug_mask}, augment_cache)

            self.cache_files.extend([original_cache, augment_cache])


    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        data = torch.load(self.cache_files[idx])
        if idx == 0:  # tylko raz
            print("Image:", data['image'].dtype, data['image'].shape, data['image'].min().item(), data['image'].max().item())
            print("Mask:", data['mask'].dtype, data['mask'].shape, torch.unique(data['mask']))
        return data['image'], data['mask']
