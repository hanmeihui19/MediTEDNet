import os
import random
import shutil

# Dataset root directory
data_dir = "Root directory of the dataset"
output_dir = "Directory to save results"

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create output directories for each split
for split in ["train", "val", "test"]:
    for class_name in os.listdir(data_dir):
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# Perform dataset splitting
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Get all image files in the current class directory
    images = [img for img in os.listdir(class_path) if img.endswith((".jpg", ".png"))]
    random.shuffle(images)  # Shuffle image list

    # Calculate the number of images for each split
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count

    # Split into train, validation, and test sets
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Copy images to the corresponding output directories
    for img_name in train_images:
        shutil.copy(os.path.join(class_path, img_name), os.path.join(output_dir, "train", class_name, img_name))
    for img_name in val_images:
        shutil.copy(os.path.join(class_path, img_name), os.path.join(output_dir, "val", class_name, img_name))
    for img_name in test_images:
        shutil.copy(os.path.join(class_path, img_name), os.path.join(output_dir, "test", class_name, img_name))

print("Dataset splitting completed!")
