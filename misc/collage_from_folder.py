import os
import random
from PIL import Image

# Set the folder containing the dataset images and the collage output file
image_folder = '/home/tomass/tomass/magistrs/video_annotating/pidgeon_datasets/pidgeon_images/pidgeon_vid_4/4'
output_collage = 'pidgeon_collage4.jpg'

# Parameters for the collage
collage_width = 6  # Number of images per row
collage_height = 1  # Number of images per column
image_size = 256  # Size of each cropped square image

# Get the list of all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Randomly select images for the collage
selected_images = random.sample(image_files, collage_width * collage_height)

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
for idx, image_file in enumerate(selected_images):
    img_path = os.path.join(image_folder, image_file)
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

# Save the collage to a file
collage.save(output_collage)
print(f"Collage saved as {output_collage}")
