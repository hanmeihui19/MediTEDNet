import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths, self.labels = self.get_image_paths_labels()
        self.transform = transform

    def get_image_paths_labels(self):
        image_paths = []
        labels = []
        label_map = {subdir: idx for idx, subdir in enumerate(os.listdir(self.root_dir))}  # Assign a label to each subdirectory

        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Only load images with .jpg, .png, or .jpeg extensions
                    image_paths.append(os.path.join(subdir, file))
                    label = label_map.get(os.path.basename(subdir))  # Get the label corresponding to the subdirectory
                    labels.append(label)

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def calculate_mean_std(dataloader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_images = 0

    for images, _ in dataloader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_images += batch_size

    mean /= n_images
    std /= n_images

    return mean, std


def main():
    # Replace with your dataset path
    root_dir = (r"Replace with your dataset path")

    batch_size = 64
    num_workers = 8

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean, std = calculate_mean_std(dataloader)

    print('Mean:', mean)
    print('Std:', std)


if __name__ == '__main__':
    main()

