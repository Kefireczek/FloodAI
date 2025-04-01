import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Dataset import FloodDataset

# Ustawienia
batch_size = 16  # Możesz dostosować rozmiar partii
num_workers = 4  # Liczba wątków do ładowania danych

def show_images_and_masks(data_loader, num_images=3):
    """
    Funkcja wyświetla obrazy i odpowiadające im maski w zwartym układzie
    """
    # Pobieranie danych i konwersja do numpy
    images, masks = next(iter(data_loader))
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    masks = masks.cpu().squeeze().numpy()

    # Tworzenie siatki wykresów
    fig, axs = plt.subplots(num_images, 2, figsize=(10, 3*num_images),
                            gridspec_kw={'wspace':0.05, 'hspace':0.1},
                            squeeze=False)

    # Ukrywanie nieużywanych osi
    for ax in axs.flat:
        ax.axis('off')

    # Wyświetlanie par obraz-maska
    for i in range(num_images):
        # Obraz
        axs[i, 0].imshow(images[i])
        axs[i, 0].set_title(f'Obraz {i+1}', pad=5)

        # Maska
        mask_plot = axs[i, 1].imshow(masks[i], cmap='gray', vmin=0, vmax=1)
        axs[i, 1].set_title(f'Maska {i+1}', pad=5)

        # Dodanie paska kolorów tylko dla masek
        if i == num_images-1:  # tylko dla ostatniego wiersza
            cbar = fig.colorbar(mask_plot, ax=axs[i, 1], shrink=0.8)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Tło', 'Powódź'])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Tworzenie instancji datasetu
    train_dataset = FloodDataset(base_folder='train_data', transform=None)

    # Tworzenie DataLoadera
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Losowe permutowanie danych
        num_workers=num_workers
    )

    # Testowanie DataLoadera (sprawdzamy rozmiar pierwszej partii)
    for i, (images, masks) in enumerate(train_loader):
        print(f'Batch {i + 1}:')
        print(f'  Obrazy: {images.size()}')
        print(f'  Maski: {masks.size()}')
        break  # Wyświetlamy tylko pierwszy batch

    # Wywołanie funkcji do wyświetlenia obrazów i masek
    show_images_and_masks(train_loader, num_images=3)
