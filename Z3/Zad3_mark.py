import os
import random
import numpy as np
import pandas as pd
from skimage import io
from skimage.feature import graycomatrix, graycoprops

# Parametry do analizy tekstur
distances = [1, 3, 5]                                                                               # Odległości do analizy GLCM
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]                                                           # Kąty do analizy GLCM
props = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']                # Właściwości GLCM do obliczenia:   # - dissimilarity: różnica między pikselami
                                                                                                                                        # - correlation: korelacja między pikselami
                                                                                                                                        # - contrast: kontrast między pikselami
                                                                                                                                        # - energy: energia tekstury
                                                                                                                                        # - homogeneity: jednorodność tekstury
                                                                                                                                        # - ASM: średni kwadrat tekstury (Angular Second Moment)

# Funkcja do ekstrakcji cech z obrazu
def extract_features(image_path, distances, angles):
    image = io.imread(image_path)                                                                   # Wczytanie obrazu
    image = (image * 255).astype(np.uint8)                                                          # Konwersja obrazu do formatu uint8

    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True) # Obliczenie macierzy GLCM
    feature_vector = []                                                                             # Inicjalizacja wektora cech
    for prop in props:                                                                              # Iteracja po właściwościach GLCM
        values = graycoprops(glcm, prop)                                                            # Obliczenie wartości właściwości GLCM
        feature_vector.extend(values.flatten())                                                     # Dodanie wartości właściwości do wektora cech
    return feature_vector                                                                           # Zwrócenie wektora cech

# Funkcja do przetwarzania zbioru danych tekstur
def process_texture_dataset(dataset_folder, max_samples_per_class):
    rows = []                                                                                       # Inicjalizacja listy do przechowywania wierszy danych
    for category in os.listdir(dataset_folder):                                                     # Iteracja po kategoriach w folderze zbioru danych
        category_path = os.path.join(dataset_folder, category)                                      # Pełna ścieżka do folderu kategorii
        if not os.path.isdir(category_path):                                                        # Sprawdzenie, czy ścieżka jest folderem
            continue                                                                                # Jeśli nie jest folderem, pomiń tę kategorię

        all_files = [f for f in os.listdir(category_path)                                           # Pobranie wszystkich plików w folderze kategorii
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]                      # Obsługiwane rozszerzenia plików
        selected_files = random.sample(all_files, min(len(all_files), max_samples_per_class))       # Losowe wybranie próbek z kategorii, maksymalnie max_samples_per_class

        for filename in selected_files:                                                             # Iteracja po wybranych plikach
            filepath = os.path.join(category_path, filename)                                        # Pełna ścieżka do pliku
            print(f"Utworzono: {filename}")                                                         # Komunikat o przetwarzaniu pliku
            features = extract_features(filepath, distances, angles)                                # Ekstrakcja cech z obrazu
            features.append(category)                                                               # Dodanie kategorii do wektora cech
            rows.append(features)                                                                   # Dodanie wektora cech do listy wierszy

    return rows                                                                                     # Zwrócenie listy wierszy danych

# Funkcja do tworzenia nagłówków dla DataFrame
def create_headers():
    headers = []                                                                                    # Inicjalizacja listy nagłówków
    for prop in props:                                                                              # Iteracja po właściwościach GLCM
        for d in distances:                                                                         # Iteracja po odległościach
            for a_deg in [0, 45, 90, 135]:                                                          # Iteracja po kątach w stopniach
                headers.append(f"{prop}_d{d}_a{a_deg}")                                             # Dodanie nagłówka dla właściwości GLCM, odległości i kąta
    headers.append("category")                                                                      # Dodanie nagłówka dla kategorii
    return headers                                                                                  # Zwrócenie listy nagłówków

# Główna część programu
if __name__ == "__main__":
    dataset_folder = input("Podaj ścieżkę do folderów z przygotowanymi teksturami: ").strip()       # Pobranie ścieżki do folderu z teksturami
    max_samples_per_class = int(input("Podaj liczbę próbek do przetworzenia: "))                    # Pobranie liczby próbek do przetworzenia
    data = process_texture_dataset(dataset_folder, max_samples_per_class)                           # Przetwarzanie zbioru danych tekstur
    headers = create_headers()                                                                      # Utworzenie nagłówków dla DataFrame
    df = pd.DataFrame(data, columns=headers)                                                        # Utworzenie DataFrame z danymi i nagłówkami
    df.to_csv("vectors_" + str(max_samples_per_class) + ".csv", index=False)                        # Zapisanie DataFrame do pliku CSV
    print("Zapisano wektory cech do pliku vectors_" + str(max_samples_per_class) + ".csv")          # Komunikat o zakończeniu przetwarzania