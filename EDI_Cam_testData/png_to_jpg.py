import os
from PIL import Image

def convert_images_to_png(directory):
    """Konvertē visus attēlus direktorijā uz PNG formātu."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Pārlūkojiet visus failus direktorijā
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Pārbaudīt, vai ir fails
        if os.path.isfile(file_path):
            try:
                # Atveriet attēlu
                with Image.open(file_path) as img:
                    # Saglabājiet attēlu PNG formātā
                    new_file_path = os.path.splitext(file_path)[0] + '.png'
                    img.convert('RGB').save(new_file_path, 'PNG')
                    print(f"Converted {file_path} to {new_file_path}")

                    # Ja nepieciešams, varat izdzēst oriģinālo failu pēc konvertēšanas
                    # os.remove(file_path)

            except Exception as e:
                print(f"Failed to convert {file_path}: {e}")

# Norādiet direktoriju, kurā atrodas attēli
image_directory = '/home/anzelika/Documents/licence_plate_detection/Kameras/EDI_Cam_testData/summer_vid/vid1/cam2'
convert_images_to_png(image_directory)
