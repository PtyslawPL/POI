import numpy as np

def generate_hplane(x_range, y_range, num_points):
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    z = np.zeros(num_points)
    return np.vstack((x, y, z)).T

def generate_vplane(x_range, z_range, num_points):
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.zeros(num_points)
    z = np.random.uniform(z_range[0], z_range[1], num_points)
    return np.vstack((x, y, z)).T

def generate_cylinder(radius, height, num_points):
    theta = np.random.uniform(0, 2*np.pi, num_points)
    z = np.random.uniform(0, height, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.vstack((x, y, z)).T

def generate_all(x_range, y_range, z_range, radius, height, num_points):
    x_hp = np.random.uniform(x_range[0], x_range[1], num_points)
    y_hp = np.random.uniform(y_range[0], y_range[1], num_points)
    z_hp = np.zeros(num_points)
    points_horizontal = np.vstack((x_hp, y_hp, z_hp)).T

    x_vp = np.random.uniform(x_range[0], x_range[1], num_points)
    z_vp = np.random.uniform(z_range[0], z_range[1], num_points)
    y_vp = np.zeros(num_points)
    points_vertical = np.vstack((x_vp, y_vp, z_vp)).T

    theta = np.random.uniform(0, 2*np.pi, num_points)
    z_cyl = np.random.uniform(-height, height, num_points)
    x_cyl = radius * np.cos(theta)
    y_cyl = radius * np.sin(theta)
    points_cylinder = np.vstack((x_cyl, y_cyl, z_cyl)).T

    return np.vstack((points_horizontal, points_vertical, points_cylinder))


def save_to_xyz(filename, points):
    np.savetxt(filename, points, fmt='%.6f')

if __name__ == "__main__":
    num_points = 1000

    points_horizontal = generate_hplane(x_range=(-10, 10), y_range=(-10, 10), num_points=num_points)
    save_to_xyz('horizontal_plane.xyz', points_horizontal)

    points_vertical = generate_vplane(x_range=(-10, 10), z_range=(-10, 10), num_points=num_points)
    save_to_xyz('vertical_plane.xyz', points_vertical)

    points_cylinder = generate_cylinder(radius=5, height=10, num_points=num_points)
    save_to_xyz('cylinder_surface.xyz', points_cylinder)

    points_all = generate_all(x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10), radius=5, height=10, num_points=num_points)
    save_to_xyz('all_points.xyz', points_all)