import numpy as np

# Funkcja generująca płaszczyznę poziomą w przestrzeni 3D, gdzie z jest stałe (z = 0).
def generate_hplane(x_range, y_range, num_points):
    x = np.random.uniform(x_range[0], x_range[1], num_points)                       # Generowanie losowych wartości x w zadanym zakresie
    y = np.random.uniform(y_range[0], y_range[1], num_points)                       # Generowanie losowych wartości y w zadanym zakresie
    z = np.zeros(num_points)                                                        # Płaszczyzna pozioma ma stałą wartość z = 0
    return np.vstack((x, y, z)).T                                                   # Łączenie x, y, z w macierz punktów 3D i transponowanie

#  Funkcja generująca płaszczyznę pionową w przestrzeni 3D, gdzie y jest stałe (y = 0).
def generate_vplane(x_range, z_range, num_points):
    x = np.random.uniform(x_range[0], x_range[1], num_points)                       # Generowanie losowych wartości x w zadanym zakresie
    y = np.zeros(num_points)                                                        # Płaszczyzna pionowa ma stałą wartość y = 0
    z = np.random.uniform(z_range[0], z_range[1], num_points)                       # Generowanie losowych wartości z w zadanym zakresie
    return np.vstack((x, y, z)).T                                                   # Łączenie x, y, z w macierz punktów 3D i transponowanie

# Funkcja generująca powierzchnię walca w przestrzeni 3D, gdzie promień i wysokość są zadane.
def generate_cylinder(radius, height, num_points):
    theta = np.random.uniform(0, 2*np.pi, num_points)                               # Generowanie losowych kątów theta w zakresie od 0 do 2*pi
    z = np.random.uniform(0, height, num_points)                                    # Generowanie losowych wartości z w zakresie od 0 do wysokości walca
    x = radius * np.cos(theta)                                                      # Obliczanie współrzędnych x na podstawie promienia i kąta theta
    y = radius * np.sin(theta)                                                      # Obliczanie współrzędnych y na podstawie promienia i kąta theta
    return np.vstack((x, y, z)).T                                                   # Łączenie x, y, z w macierz punktów 3D i transponowanie

# Funkcja generująca wszystkie trzy typy punktów: płaszczyznę poziomą, płaszczyznę pionową i powierzchnię walca.
def generate_all(x_range, y_range, z_range, radius, offset, height, num_points):
    x_hp = np.random.uniform(x_range[0] - offset, x_range[1] - offset, num_points)  # Generowanie losowych wartości x w zadanym zakresie z przesunięciem
    y_hp = np.random.uniform(y_range[0], y_range[1], num_points)                    # Generowanie losowych wartości y w zadanym zakresie
    z_hp = np.zeros(num_points)                                                     # Płaszczyzna pozioma ma stałą wartość z = 0
    points_horizontal = np.vstack((x_hp, y_hp, z_hp)).T                             # Łączenie x, y, z w macierz punktów 3D i transponowanie

    x_vp = np.random.uniform(x_range[0] + offset, x_range[1] + offset, num_points)  # Generowanie losowych wartości x w zadanym zakresie z przesunięciem
    z_vp = np.random.uniform(z_range[0], z_range[1], num_points)                    # Generowanie losowych wartości z w zadanym zakresie
    y_vp = np.zeros(num_points)                                                     # Płaszczyzna pionowa ma stałą wartość y = 0
    points_vertical = np.vstack((x_vp, y_vp, z_vp)).T                               # Łączenie x, y, z w macierz punktów 3D i transponowanie

    theta = np.random.uniform(0, 2*np.pi, num_points)                               # Generowanie losowych kątów theta w zakresie od 0 do 2*pi
    z_cyl = np.random.uniform(-height, height, num_points)                          # Generowanie losowych wartości z w zakresie od -wysokości do wysokości walca
    x_cyl = radius * np.cos(theta)                                                  # Obliczanie współrzędnych x na podstawie promienia i kąta theta
    y_cyl = radius * np.sin(theta)                                                  # Obliczanie współrzędnych y na podstawie promienia i kąta theta
    points_cylinder = np.vstack((x_cyl, y_cyl, z_cyl)).T                            # Łączenie x, y, z w macierz punktów 3D i transponowanie

    return np.vstack((points_horizontal, points_vertical, points_cylinder))         # Łączenie wszystkich punktów w jedną macierz

# Funkcja do zapisywania punktów do pliku w formacie XYZ.
def save_to_xyz(filename, points):                                                  # Funkcja do zapisywania punktów do pliku w formacie XYZ
    np.savetxt(filename, points, fmt='%.6f')                                        # Zapis punktów do pliku z formatowaniem do 6 miejsc po przecinku

# Główna funkcja, która generuje punkty i zapisuje je do plików.
if __name__ == "__main__":
    num_points = 1000                                                               # Liczba punktów do wygenerowania dla każdej płaszczyzny i walca

    # Generowanie punktów dla płaszczyzny poziomej i zapisywanie ich do plików
    points_horizontal = generate_hplane(x_range=(-10, 10), y_range=(-10, 10), num_points=num_points)
    save_to_xyz('horizontal_plane.xyz', points_horizontal)

    # Generowanie punktów dla płaszczyzny pionowej i zapisywanie ich do pliku
    points_vertical = generate_vplane(x_range=(-10, 10), z_range=(-10, 10), num_points=num_points)
    save_to_xyz('vertical_plane.xyz', points_vertical)

    # Generowanie punktów dla powierzchni walca i zapisywanie ich do pliku
    points_cylinder = generate_cylinder(radius=5, height=10, num_points=num_points)
    save_to_xyz('cylinder_surface.xyz', points_cylinder)

    # Generowanie wszystkich punktów (płaszczyzny poziomej, pionowej i walca) i zapisywanie ich do pliku
    points_all = generate_all(x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10), offset = 20, radius=5, height=10, num_points=num_points)
    save_to_xyz('all_points.xyz', points_all)