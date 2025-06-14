import numpy as np
import random
import csv
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from pyransac3d import Plane

# Wczytanie danych z pliku XYZ
def load_xyz(filename):
    points = []                                                                         # Lista do przechowywania punktów
    with open(filename, 'r') as file:                                                   # Otwarcie pliku
        reader = csv.reader(file, delimiter=' ')                                        # Wczytanie danych z pliku CSV
        for row in reader:                                                              # Iteracja po wierszach
            if len(row) == 3:                                                           # Sprawdzenie, czy wiersz zawiera dokładnie 3 współrzędne
                points.append([float(coord) for coord in row])                          # Konwersja współrzędnych do float i dodanie do listy
    return np.array(points)                                                             # Zwrócenie punktów jako tablicy NumPy

# Dopasowanie płaszczyzny do punktów za pomocą RANSAC
def fit_plane_ransac(points, threshold=2.5, max_iterations=1000):                       # Maksymalna liczba iteracji
    best_eq = None                                                                      # Wyzerowanie najlepszego dopasowania płaszczyzny
    best_inliers = []                                                                   # Wyzerowanie najlepszych punktów wewnętrznych

    for _ in range(max_iterations):                                                     # Pętla do maksymalnej liczby iteracji
        sample_indices = random.sample(range(points.shape[0]), 3)                       # Losowanie 3 punktów z danych
        p1, p2, p3 = points[sample_indices]                                             # Wybranie punktów z wylosowanych indeksów

        normal = np.cross(p2 - p1, p3 - p1)                                             # Obliczenie wektora normalnego płaszczyzny
        if np.linalg.norm(normal) == 0:                                                 # Sprawdzenie, czy wektor normalny jest zerowy
            continue                                                                    # Jeśli tak, pomiń tę iterację

        normal = normal / np.linalg.norm(normal)                                        # Normalizacja wektora normalnego
        A, B, C = normal                                                                # Współrzędne wektora normalnego
        D = -np.dot(normal, p1)                                                         # Obliczenie D z równania płaszczyzny Ax + By + Cz + D = 0

        distances = np.abs(A * points[:,0] + B * points[:,1] + C * points[:,2] + D)     # Obliczenie odległości punktów od płaszczyzny
        inliers = np.where(distances < threshold)[0]                                    # Znalezienie punktów wewnętrznych (inliers)

        if len(inliers) > len(best_inliers):                                            # Sprawdzenie, czy znaleziono więcej punktów wewnętrznych niż dotychczas
            best_inliers = inliers                                                      # Zaktualizowanie najlepszych punktów wewnętrznych
            best_eq = (A, B, C, D)                                                      # Zaktualizowanie najlepszego dopasowania płaszczyzny

    return best_eq, best_inliers                                                        # Zwrócenie najlepszego dopasowania płaszczyzny i punktów wewnętrznych

# Dopasowanie płaszczyzny do punktów za pomocą Pyransac
def fit_plane_pyransac(points):
    plane = Plane()                                                                     # Inicjalizacja obiektu Plane z biblioteki Pyransac
    a, b, c, d, inliers = plane.fit(points, thresh=2.5)                                 # Dopasowanie płaszczyzny do punktów z progiem 2.5
    return (a, b, c, d), inliers                                                        # Zwrócenie współczynników płaszczyzny i punktów wewnętrznych


# Klasteryzacja punktów za pomocą KMeans
def cluster_points_kmeans(points, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)                                      # Inicjalizacja KMeans z liczbą klastrów k
    labels = kmeans.fit_predict(points)                                                 # Dopasowanie modelu KMeans do punktów i uzyskanie etykiet klastrów
    clusters = [points[labels == i] for i in range(k)]                                  # Podział punktów na klastry na podstawie etykiet
    for i, cluster in enumerate(clusters):                                              # Iteracja po klastrach
        print(f"Klaster {i}: liczba punktów = {len(cluster)}")                          # Wyświetlenie liczby punktów w klastrze
    return clusters                                                                     # Zwrócenie listy klastrów, gdzie każdy klaster to tablica punktów

# Klasteryzacja punktów za pomocą DBSCAN
def cluster_points_dbscan(points, eps=4.0, min_samples=50):
    db = DBSCAN(eps=eps, min_samples=min_samples)                                       # Inicjalizacja DBSCAN z parametrami eps i min_samples
    labels = db.fit_predict(points)                                                     # Dopasowanie modelu DBSCAN do punktów i uzyskanie etykiet klastrów
    clusters = []                                                                       # Lista do przechowywania klastrów
    for label in np.unique(labels):                                                     # Iteracja po unikalnych etykietach klastrów
        if label != -1:                                                                 # Sprawdzenie, czy etykieta nie jest -1 (punkty szumowe)
            clusters.append(points[labels == label])                                    # Dodanie punktów z danym etykietą do listy klastrów
    return clusters                                                                     # Zwrócenie listy klastrów, gdzie każdy klaster to tablica punktów

# Analiza klastrów i dopasowanie płaszczyzny z użyciem dopasowania ransac
def analyze_clusters_r(clusters):
    for idx, cluster in enumerate(clusters):                                            # Iteracja po klastrach
        eq, inliers = fit_plane_ransac(cluster)                                         # Dopasowanie płaszczyzny do punktów w klastrze

        if eq is not None:                                                              # Sprawdzenie, czy udało się dopasować płaszczyznę
            A, B, C, D = eq                                                             # Rozpakowanie współczynników płaszczyzny
            normal_vector = np.array([A, B, C])                                         # Utworzenie wektora normalnego płaszczyzny
            normal_vector = normal_vector / np.linalg.norm(normal_vector)               # Normalizacja wektora normalnego

            distances = np.abs(A * cluster[:,0] + B * cluster[:,1] + C * cluster[:,2] + D) / np.linalg.norm([A, B, C]) # Obliczenie odległości punktów od płaszczyzny
            mean_distance = np.mean(distances)                                          # Obliczenie średniej odległości punktów od płaszczyzny

            print(f"Chmura {idx+1}:")                                                   # Wyświetlenie numeru chmury
            print(f"  Wektor normalny: {normal_vector}")                                # Wyświetlenie wektora normalnego płaszczyzny
            print(f"  Średnia odległość: {mean_distance:.6f}")                          # Wyświetlenie średniej odległości punktów od płaszczyzny

            if mean_distance < 0.01:                                                    # Sprawdzenie, czy średnia odległość jest mniejsza niż 0.01
                print("  -> To jest płaszczyzna.")                                      # Wyświetlenie komunikatu, że to jest płaszczyzna
                if abs(C) > 0.9:                                                        # Dominująca oś Z
                    print("  -> Płaszczyzna pozioma (dominująca oś Z).")                # Wyświetlenie komunikatu o płaszczyźnie poziomej
                elif abs(B) > 0.9:                                                      # Dominująca oś Y
                    print("  -> Płaszczyzna pionowa (dominująca oś Y).")                # Wyświetlenie komunikatu o płaszczyźnie pionowej
                elif abs(A) > 0.9:                                                      # Dominująca oś X
                    print("  -> Płaszczyzna pionowa (dominująca oś X).")                # Wyświetlenie komunikatu o płaszczyźnie pionowej
                else:                                                                   # Płaszczyzna pod kątem
                    print("  -> Płaszczyzna pod kątem.")                                # Wyświetlenie komunikatu o płaszczyźnie pod kątem
            else:                                                                       # Jeśli średnia odległość jest większa niż 0.01
                print("  -> To nie jest płaszczyzna.")                                  # Wyświetlenie komunikatu, że to nie jest płaszczyzna
        else:                                                                           # Jeśli nie udało się dopasować płaszczyzny
            print(f"Chmura {idx+1}: Nie udało się dopasować płaszczyzny.")              # Wyświetlenie komunikatu o braku dopasowania płaszczyzny

# Analiza klastrów i dopasowanie płaszczyzny z użyciem dopasowania pyransac
def analyze_clusters_p(clusters):
    for idx, cluster in enumerate(clusters):                                            # Iteracja po klastrach
        eq, inliers = fit_plane_pyransac(cluster)                                       # Dopasowanie płaszczyzny do punktów w klastrze

        if eq is not None:                                                              # Sprawdzenie, czy udało się dopasować płaszczyznę
            A, B, C, D = eq                                                             # Rozpakowanie współczynników płaszczyzny
            normal_vector = np.array([A, B, C])                                         # Utworzenie wektora normalnego płaszczyzny
            normal_vector = normal_vector / np.linalg.norm(normal_vector)               # Normalizacja wektora normalnego

            distances = np.abs(A * cluster[:,0] + B * cluster[:,1] + C * cluster[:,2] + D) / np.linalg.norm([A, B, C]) # Obliczenie odległości punktów od płaszczyzny
            mean_distance = np.mean(distances)                                          # Obliczenie średniej odległości punktów od płaszczyzny

            print(f"Chmura {idx+1}:")                                                   # Wyświetlenie numeru chmury
            print(f"  Wektor normalny: {normal_vector}")                                # Wyświetlenie wektora normalnego płaszczyzny
            print(f"  Średnia odległość: {mean_distance:.6f}")                          # Wyświetlenie średniej odległości punktów od płaszczyzny

            if mean_distance < 0.01:                                                    # Sprawdzenie, czy średnia odległość jest mniejsza niż 0.01
                print("  -> To jest płaszczyzna.")                                      # Wyświetlenie komunikatu, że to jest płaszczyzna
                if abs(C) > 0.9:                                                        # Dominująca oś Z
                    print("  -> Płaszczyzna pozioma (dominująca oś Z).")                # Wyświetlenie komunikatu o płaszczyźnie poziomej
                elif abs(B) > 0.9:                                                      # Dominująca oś Y
                    print("  -> Płaszczyzna pionowa (dominująca oś Y).")                # Wyświetlenie komunikatu o płaszczyźnie pionowej
                elif abs(A) > 0.9:                                                      # Dominująca oś X
                    print("  -> Płaszczyzna pionowa (dominująca oś X).")                # Wyświetlenie komunikatu o płaszczyźnie pionowej
                else:                                                                   # Płaszczyzna pod kątem
                    print("  -> Płaszczyzna pod kątem.")                                # Wyświetlenie komunikatu o płaszczyźnie pod kątem
            else:                                                                       # Jeśli średnia odległość jest większa niż 0.01
                print("  -> To nie jest płaszczyzna.")                                  # Wyświetlenie komunikatu, że to nie jest płaszczyzna
        else:                                                                           # Jeśli nie udało się dopasować płaszczyzny
            print(f"Chmura {idx+1}: Nie udało się dopasować płaszczyzny.")              # Wyświetlenie komunikatu o braku dopasowania płaszczyzny

# Główna funkcja do uruchomienia programu
if __name__ == "__main__":
    filename = input("Podaj ścieżkę do pliku z danymi: ")                               # Wczytanie nazwy pliku z danymi
    xyz_file = load_xyz(filename)                                                       # Wczytanie danych z pliku XYZ

    clusters_k = cluster_points_kmeans(xyz_file, k=3)                                   # Dopasowanie liczby klastrów do 3
    print("=== ANALIZA CHMUR DLA KLASZTORÓW K-MEANS ===") 
    analyze_clusters_r(clusters_k)                                                      # Dopasowanie płaszczyzny z użyciem RANSAC

    clusters_d = cluster_points_dbscan(xyz_file)                                        # Dopasowanie liczby klastrów do DBSCAN
    print("=== ANALIZA CHMUR DLA KLASZTORÓW DBSCAN ===")
    analyze_clusters_r(clusters_d)                                                      # Dopasowanie płaszczyzny z użyciem RANSAC