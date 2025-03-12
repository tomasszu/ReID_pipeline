import matplotlib.pyplot as plt
import pandas as pd

# Read data from a text file
file_path = '/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/automated_training/veri_unmodified_corrRes.txt'  # Specify the path to your text file

# Load the data into a pandas DataFrame
df = pd.read_csv(file_path)

# Strip whitespace from headers (if necessary)
df.columns = df.columns.str.strip()

# Debugging: Print the first few rows and the columns
print(df.head())  # Check the first few rows
print(df.columns)  # Check the column names

# Separate the training and validation data
train_data = df[df['phase'] == 'train']
val_data = df[df['phase'] == 'val']

# Subtract 10 from all validation loss values
val_data['loss'] = val_data['loss'] - 10

# Plotting Loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_data['epoch'], train_data['loss'], label='Train Loss', marker='o')
plt.plot(val_data['epoch'], val_data['loss'], label='Val Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# # Plotting Accuracy
# plt.subplot(1, 2, 2)
# plt.plot(train_data['epoch'], train_data['accuracy'], label='Train Accuracy', marker='o')
# plt.plot(val_data['epoch'], val_data['accuracy'], label='Val Accuracy', marker='o')
# plt.title('Accuracy over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

plt.tight_layout()
plt.show()
