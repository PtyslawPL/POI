import os
from PIL import Image

# Parametry
patch_size = 128
supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

def process_patch(patch):
    gray = patch.convert("L")
    reduced = gray.point(lambda x: int(x / 4) * 4)
    return reduced

def slice_image(image_path, output_subfolder, patch_size):
    image = Image.open(image_path)
    img_width, img_height = image.size
    basename = os.path.splitext(os.path.basename(image_path))[0]

    os.makedirs(output_subfolder, exist_ok=True)

    num_x = img_width // patch_size
    num_y = img_height // patch_size
    total_patches = num_x * num_y

    patch_id = 0
    for y in range(0, img_height, patch_size):
        for x in range(0, img_width, patch_size):
            box = (x, y, x + patch_size, y + patch_size)
            patch = image.crop(box)

            if patch.size[0] == patch_size and patch.size[1] == patch_size:
                processed_patch = process_patch(patch)
                patch_filename = f"{basename}_patch_{patch_id}.png"
                patch_path = os.path.join(output_subfolder, patch_filename)
                processed_patch.save(patch_path)
                patch_id += 1

def process_all_images(file_path, patch_size):
    images = [f for f in os.listdir(file_path) if f.lower().endswith(supported_extensions)]
    if not images:
        print("Brak obsługiwanych zdjęć w podanym folderze.")
        return

    for filename in os.listdir(file_path):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(file_path, filename)
            basename = os.path.splitext(filename)[0]
            output_subfolder = os.path.join(file_path, basename)
            slice_image(image_path, output_subfolder, patch_size)

# Uruchomienie
file_path = input("Podaj ścieżkę do folderu ze zdjęciami: ")
process_all_images(file_path, patch_size)