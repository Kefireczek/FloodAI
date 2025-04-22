import os
import matplotlib.pyplot as plt
from PIL import Image


def extract_image_data(image_folder):
    """
    Przetwarza folder z obrazami i zwraca listę rozdzielczości oraz aspect ratio.

    :param image_folder: Ścieżka do folderu zawierającego obrazy.
    :return: Tuple (resolutions, aspect_ratios), gdzie:
             - resolutions to lista krotek (szerokość, wysokość)
             - aspect_ratios to lista wartości szerokość / wysokość
    """
    resolutions = []
    aspect_ratios = []

    for img_name in os.listdir(image_folder):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, img_name)

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    resolutions.append((width, height))
                    aspect_ratios.append(width / height)
            except Exception as e:
                print(f"Błąd otwierania {img_name}: {e}")

    return resolutions, aspect_ratios


def analyze_and_plot(resolutions, aspect_ratios):
    """
    Analizuje dane zdjęć i generuje wykresy.

    :param resolutions: Lista krotek (szerokość, wysokość).
    :param aspect_ratios: Lista wartości aspect ratio (szerokość / wysokość).
    """
    if not resolutions:
        print("Brak danych do analizy.")
        return

    # Znalezienie min/max zdjęć
    min_res = min(resolutions, key=lambda x: x[0] * x[1])
    max_res = max(resolutions, key=lambda x: x[0] * x[1])

    resolutions_mp = [(w * h) / 1_000_000 for w, h in resolutions]

    # Histogram rozdzielczości
    # Przeliczenie rozdzielczości na megapiksele
    resolutions_mp = [(w * h) / 1_000_000 for w, h in resolutions]

    # Wykorzystanie tych samych przedziałów jak wcześniej
    counts, _, _ = plt.hist(resolutions_mp, bins=20, color='blue', alpha=0.7, edgecolor='black')

    # Etykiety na osi X w megapikselach (MP)
    plt.xlabel("Rozdzielczość (MP)")
    plt.ylabel("Liczba zdjęć")
    plt.title("Histogram rozdzielczości zdjęć")
    plt.grid(True)
    plt.show()

    # Histogram aspect ratio
    plt.figure(figsize=(10, 5))
    plt.hist(aspect_ratios, bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel("Stosunek szerokość / wysokość")
    plt.ylabel("Liczba zdjęć")
    plt.title("Histogram aspect ratio")
    plt.grid(True)
    plt.show()

    # Wyświetlenie skrajnych zdjęć
    print(f"Najmniejsze zdjęcie: {min_res[0]}x{min_res[1]}")
    print(f"Największe zdjęcie: {max_res[0]}x{max_res[1]}")


def compute_positive_pixel_ratio(loader):
    total_pixels = 0
    total_positive = 0

    for _, masks in loader:
        masks = masks.view(-1)  # spłaszczamy wszystko
        total_pixels += masks.numel()
        total_positive += masks.sum().item()  # liczba "1" w maskach

    ratio = total_positive / total_pixels
    return ratio

# Główna funkcja
if __name__ == "__main__":
    folder_path = "train_data/train_images"
    # resolutions, aspect_ratios = extract_image_data(folder_path)
    # analyze_and_plot(resolutions, aspect_ratios)
