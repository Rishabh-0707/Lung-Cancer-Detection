# Install kagglehub and Pillow if not already installed
# pip install kagglehub
# pip install Pillow


import kagglehub
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Step 1: Download the lung cancer dataset from kaggle
path = kagglehub.dataset_download("adityamahimkar/iqothnccd-lung-cancer-dataset")
print("✅ Path to dataset files:", path)

# Step 2: Debug - Print dataset contents
print("📂 Contents of dataset folder:", os.listdir(path))

# Step 3: Ensure the dataset structure
expected_folders = ['cancer', 'no_cancer']
current_folders = os.listdir(path)

# If folders not there, create them (assuming labels are in filenames)
if not all(folder in current_folders for folder in expected_folders):
    print("⚠ Dataset not structured correctly. Fixing...")

    # Create cancer and no_cancer folders
    cancer_folder = os.path.join(path, 'cancer')
    no_cancer_folder = os.path.join(path, 'no_cancer')
    os.makedirs(cancer_folder, exist_ok=True)
    os.makedirs(no_cancer_folder, exist_ok=True)

    # Move files based on name (assuming the files names are 'cancer' or 'no_cancer')
    for file_name in current_folders:
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            if 'cancer' in file_name.lower():
                shutil.move(file_path, cancer_folder)
            else:
                shutil.move(file_path, no_cancer_folder)

    print("✅ Dataset structure fixed!")
    print("📂 New folder contents:", os.listdir(path))

# Step 4: Set the image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32

# Step 5: Create ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% training, 20% validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Step 6: Build a simple CNN model
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 7: Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 8: Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Step 9: Save the trained model
model.save('lung_cancer_detector.h5')
print("✅ Model training complete and saved as 'lung_cancer_detector.h5'")

# Step 10: Plotting training & validation accuracy/loss graphs
# Plot Training & Validation Accuracy
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()
