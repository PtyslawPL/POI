import os
import random
import numpy as np
import pandas as pd
from skimage import io
from skimage.feature import graycomatrix, graycoprops

distances = [1, 3, 5]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
props = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

def extract_features(image_path, distances, angles):
    image = io.imread(image_path)
    image = (image * 255).astype(np.uint8)

    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    feature_vector = []
    for prop in props:
        values = graycoprops(glcm, prop)
        feature_vector.extend(values.flatten())
    return feature_vector

def process_texture_dataset(dataset_folder, max_samples_per_class):
    rows = []
    for category in os.listdir(dataset_folder):
        category_path = os.path.join(dataset_folder, category)
        if not os.path.isdir(category_path):
            continue

        all_files = [f for f in os.listdir(category_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
        selected_files = random.sample(all_files, min(len(all_files), max_samples_per_class))

        for filename in selected_files:
            filepath = os.path.join(category_path, filename)
            print(f"Utworzono: {filename}")
            features = extract_features(filepath, distances, angles)
            features.append(category)
            rows.append(features)

    return rows

def create_headers():
    headers = []
    for prop in props:
        for d in distances:
            for a_deg in [0, 45, 90, 135]:
                headers.append(f"{prop}_d{d}_a{a_deg}")
    headers.append("category")
    return headers

if __name__ == "__main__":
    dataset_folder = input("Podaj ścieżkę do folderów z przygotowanymi teksturami: ").strip()
    max_samples_per_class = int(input("Podaj liczbę próbek do przetworzenia: "))
    data = process_texture_dataset(dataset_folder, max_samples_per_class)
    headers = create_headers()
    df = pd.DataFrame(data, columns=headers)
    df.to_csv("vectors_" + str(max_samples_per_class) + ".csv", index=False)
    print("Zapisano wektory cech do pliku vectors_" + str(max_samples_per_class) + ".csv")