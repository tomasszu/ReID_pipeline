import csv
import os
import random
from PIL import Image

# Set the base directory for images and the CSV file
data_dir = '/home/tomass/tomass/magistrs/Animal-Identification-from-Video-main'
csv_file = '/home/tomass/tomass/magistrs/Animal-Identification-from-Video-main/Pigeons_29033_960_540_300f_train.csv'
output_collage = 'pidgeon_collage3.jpg'

# Parameters for the collage
collage_width = 4  # Number of images per row
collage_height = 5  # Number of images per column
image_size = 256  # Size of each cropped square image

# Load image paths from the CSV (relative to data_dir)
image_paths = []
with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if row:
            relative_path = row[0]
            full_path = os.path.join(data_dir, relative_path)
            image_paths.append(full_path)

# Randomly select images for the collage
selected_images = random.sample(image_paths, collage_width * collage_height)

# Create a blank image for the collage
collage = Image.new('RGB', (collage_width * image_size, collage_height * image_size))

# Function to crop the image to a square
def crop_to_square(img):
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = (width + min_dim) // 2
    bottom = (height + min_dim) // 2
    return img.crop((left, top, right, bottom))

# Add each selected image to the collage
for idx, img_path in enumerate(selected_images):
    try:
        img = Image.open(img_path)

        # Crop to square and resize
        img = crop_to_square(img)
        img = img.resize((image_size, image_size))

        # Calculate the position of the image in the collage
        row = idx // collage_width
        col = idx % collage_width
        x_pos = col * image_size
        y_pos = row * image_size

        # Paste the image into the collage
        collage.paste(img, (x_pos, y_pos))
    except Exception as e:
        print(f"Failed to load image {img_path}: {e}")

# Save the collage to a file
collage.save(output_collage)
print(f"Collage saved as {output_collage}")
