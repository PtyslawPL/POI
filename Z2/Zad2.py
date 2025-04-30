import numpy as np
import random
import csv
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from pyransac3d import Plane

def load_xyz(filename):
    points = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            if len(row) == 3:
                points.append([float(coord) for coord in row])
    return np.array(points)

def fit_plane_ransac(points, threshold=2.5, max_iterations=1000):
    best_eq = None
    best_inliers = []

    for _ in range(max_iterations):
        sample_indices = random.sample(range(points.shape[0]), 3)
        p1, p2, p3 = points[sample_indices]

        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) == 0:
            continue

        normal = normal / np.linalg.norm(normal)
        A, B, C = normal
        D = -np.dot(normal, p1)

        distances = np.abs(A * points[:,0] + B * points[:,1] + C * points[:,2] + D)
        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_eq = (A, B, C, D)

    return best_eq, best_inliers

def cluster_points_kmeans(points, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(points)
    clusters = [points[labels == i] for i in range(k)]
    for i, cluster in enumerate(clusters):
        print(f"Klaster {i}: liczba punktów = {len(cluster)}")
    return clusters

def analyze_clusters(clusters):
    for idx, cluster in enumerate(clusters):
        eq, inliers = fit_plane_ransac(cluster)

        if eq is not None:
            A, B, C, D = eq
            normal_vector = np.array([A, B, C])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)

            distances = np.abs(A * cluster[:,0] + B * cluster[:,1] + C * cluster[:,2] + D) / np.linalg.norm([A, B, C])
            mean_distance = np.mean(distances)

            print(f"Chmura {idx+1}:")
            print(f"  Wektor normalny: {normal_vector}")
            print(f"  Średnia odległość: {mean_distance:.6f}")

            if mean_distance < 0.01:
                print("  -> To jest płaszczyzna.")
                if abs(C) > 0.9:
                    print("  -> Płaszczyzna pozioma (dominująca oś Z).")
                elif abs(B) > 0.9:
                    print("  -> Płaszczyzna pionowa (dominująca oś Y).")
                elif abs(A) > 0.9:
                    print("  -> Płaszczyzna pionowa (dominująca oś X).")
                else:
                    print("  -> Płaszczyzna pod kątem.")
            else:
                print("  -> To nie jest płaszczyzna.")
        else:
            print(f"Chmura {idx+1}: Nie udało się dopasować płaszczyzny.")

def cluster_points_dbscan(points, eps=2.0, min_samples=50):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    clusters = []
    for label in np.unique(labels):
        if label != -1:
            clusters.append(points[labels == label])
    return clusters

def fit_plane_pyransac(points):
    plane = Plane()
    a, b, c, d, inliers = plane.fit(points, thresh=2.5)
    return (a, b, c, d), inliers

if __name__ == "__main__":
    filename = input("Podaj ścieżkę do pliku z danymi: ")
    xyz_file = load_xyz(filename)

    clusters = cluster_points_kmeans(xyz_file, k=3)

    print("=== ANALIZA CHMUR ===")
    analyze_clusters(clusters)
