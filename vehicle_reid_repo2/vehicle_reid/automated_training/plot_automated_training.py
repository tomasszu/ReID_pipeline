import matplotlib.pyplot as plt
import pandas as pd

# Read data from a text file
file_path = '/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/automated_training/veri_unmodified.txt'  # Specify the path to your text file

# Load the data into a pandas DataFrame
df = pd.read_csv(file_path)

# Separate the training and validation data
train_data = df[df['phase'] == 'train']
val_data = df[df['phase'] == 'val']

# Plotting Loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_data['epoch'], train_data['loss'], label='Train Loss', marker='o')
plt.plot(val_data['epoch'], val_data['loss'], label='Val Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_data['epoch'], train_data['accuracy'], label='Train Accuracy', marker='o')
plt.plot(val_data['epoch'], val_data['accuracy'], label='Val Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
