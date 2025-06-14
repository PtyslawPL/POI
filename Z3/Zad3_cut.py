import os
from PIL import Image

# Parametry
patch_size = 128                                                                # Rozdzielczość fragmentów
supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")               # Obsługiwane rozszerzenia plików

# Funkcje do przetwarzania obrazów
def process_patch(patch):
    gray = patch.convert("L")                                                   # Konwersja do skali szarości
    reduced = gray.point(lambda x: int(x / 4) * 4)                              # Redukcja wartości pikseli do wielokrotności 4
    return reduced

# Funkcja do dzielenia obrazu na fragmenty
def slice_image(image_path, output_subfolder, patch_size):
    image = Image.open(image_path)                                              # Wczytanie obrazu
    img_width, img_height = image.size                                          # Pobranie rozmiaru obrazu
    basename = os.path.splitext(os.path.basename(image_path))[0]                # Pobranie nazwy pliku bez rozszerzenia

    os.makedirs(output_subfolder, exist_ok=True)                                # Utworzenie podfolderu, jeśli nie istnieje

    num_x = img_width // patch_size                                             # Obliczenie liczby fragmentów w poziomie
    num_y = img_height // patch_size                                            # Obliczenie liczby fragmentów w pionie

    patch_id = 0                                                                # Inicjalizacja identyfikatora fragmentu
    for y in range(0, img_height, patch_size):                                  # Iteracja po wierszach
        for x in range(0, img_width, patch_size):                               # Iteracja po kolumnach
            box = (x, y, x + patch_size, y + patch_size)                        # Definicja obszaru fragmentu
            patch = image.crop(box)                                             # Wycięcie fragmentu z obrazu

            if patch.size[0] == patch_size and patch.size[1] == patch_size:     # Sprawdzenie, czy fragment ma odpowiedni rozmiar
                processed_patch = process_patch(patch)                          # Zapisanie przetworzonego fragmentu
                patch_filename = f"{basename}_patch_{patch_id}.png"             # Nazwa pliku fragmentu
                patch_path = os.path.join(output_subfolder, patch_filename)     # Ścieżka do pliku fragmentu
                processed_patch.save(patch_path)                                # Zapisanie przetworzonego fragmentu
                patch_id += 1                                                   # Zwiększenie identyfikatora fragmentu

# Funkcja do przetwarzania wszystkich obrazów w folderze
def process_all_images(file_path, patch_size):
    images = [f for f in os.listdir(file_path) if f.lower().endswith(supported_extensions)] # Lista obsługiwanych obrazów
    if not images:                                                              # Sprawdzenie, czy są jakieś obsługiwane obrazy
        print("Brak obsługiwanych zdjęć w podanym folderze.")                   # Komunikat, jeśli brak obrazów
        return

    for filename in os.listdir(file_path):                                      # Iteracja po plikach w folderze
        if filename.lower().endswith(supported_extensions):                     # Sprawdzenie, czy plik ma obsługiwane rozszerzenie
            image_path = os.path.join(file_path, filename)                      # Pełna ścieżka do pliku obrazu
            basename = os.path.splitext(filename)[0]                            # Nazwa pliku bez rozszerzenia
            print(f"Przetwarzanie obrazu: {filename}")                          # Komunikat o przetwarzaniu obrazu
            output_subfolder = os.path.join(file_path, basename)                # Podfolder dla przetworzonych fragmentów
            slice_image(image_path, output_subfolder, patch_size)               # Wywołanie funkcji do dzielenia obrazu na fragmenty

# Główna część programu
file_path = input("Podaj ścieżkę do folderu ze zdjęciami: ") .strip()           # Pobranie ścieżki do folderu ze zdjęciami
process_all_images(file_path, patch_size)                                       # Wywołanie funkcji do przetwarzania wszystkich obrazów w folderze