import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis
import torchvision.transforms.functional as TF
import numpy as np
from datetime import datetime

from utils.metrics import metrics_cal_multiclass
from utils.utils import create_lr_scheduler,EarlyStopping,seed

from Model import VSSM as MediTEDNet # import model

# set random seed
seed(0)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # Data transformation for test set
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(root="the path of your test set",
                                        transform=data_transform)
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=nw,
                                              worker_init_fn=np.random.seed(0))

    print("Using {} images for testing.".format(test_num))

    # Load model and send to device
    num_classes = len(test_dataset.classes)
    net = MediTEDNet(num_classes=num_classes)
    model_path = './{}Net.pth'.format(model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist. Please check the path.")

    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    net.eval()
    print("Model loaded successfully from {}".format(model_path))

    all_preds = []
    all_labels = []
    all_probs = []

    total_accuracy = 0.0

    misclassified_dir = "./misclassified_samples"
    os.makedirs(misclassified_dir, exist_ok=True)
    misclassified_info = []  # Store information of misclassified images

    all_image_info = []  # Store prediction probability information for each image
    class_names = test_dataset.classes  # Class names

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            predict_y = torch.max(outputs, dim=1)[1]
            probs = torch.softmax(outputs, dim=1)

            # Get the original image path
            image_path, _ = test_dataset.imgs[idx]

            # Convert softmax probability tensor to a regular list
            probs_np = probs.squeeze().cpu().numpy().tolist()

            # Construct a class-to-probability mapping
            probs_dict = {
                class_names[i]: float(prob) for i, prob in enumerate(probs_np)
            }

            # Record the prediction information for the current image
            all_image_info.append({
                "index": idx,
                "image_path": image_path,
                "true_class": class_names[labels.item()],
                "predicted_class": class_names[predict_y.item()],
                "probs": probs_dict
            })

            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total_accuracy += (predict_y == labels).sum().item()

            # === If the prediction is incorrect, record and save the image ===
            if predict_y.item() != labels.item():
                img_tensor = images.cpu().squeeze(0)
                class_names = test_dataset.classes
                pred_label = class_names[predict_y.item()]
                true_label = class_names[labels.item()]

                print(f"[MISCLASSIFIED] Index: {idx}, Pred: {pred_label}, True: {true_label}")

                # Restore image pixel values for visualization (denormalization)
                mean = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
                std = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
                img_denorm = img_tensor * std + mean

                save_path = os.path.join(misclassified_dir,
                                         f"misclassified_{idx}_pred_{pred_label}_true_{true_label}.png")
                TF.to_pil_image(img_denorm).save(save_path)

                save_path = os.path.join(
                    misclassified_dir,
                    f"misclassified_{idx}_pred_{pred_label}_true_{true_label}.png"
                )
                TF.to_pil_image(img_denorm).save(save_path)

                # Record information to the list
                misclassified_info.append({
                    "index": idx,
                    "image_path": save_path,
                    "predicted_class": pred_label,
                    "true_class": true_label
                })

    # Calculate accuracy
    accuracy = total_accuracy / test_num

    # Calculate multi-class metrics
    conf_matrix, precision, recall, f1, specificity, auc, per_class_metrics = metrics_cal_multiclass(all_labels,
                                                                                                     all_preds,
                                                                                                     all_probs)

    # Log results
    log_data = {
        'test_accurate': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'test_auc': float(auc),
        'test_specificity': float(specificity),
        'conf_matrix': conf_matrix.tolist(),
        'per_class_metrics': per_class_metrics
    }

    # Print results to console
    print(
          f"test_accurate: {log_data['test_accurate']:.3f} "
          f"test_precision: {log_data['test_precision']:.3f} "
          f"test_recall: {log_data['test_recall']:.3f} "
          f"test_f1: {log_data['test_f1']:.3f} "
          f"test_auc: {log_data['test_auc']:.3f} "
          f"test_specificity: {log_data['test_specificity']:.3f} "
          )

    log_file = './{}Net.json'.format(model_name)

    # Save misclassification information to a JSON file
    misclassified_log_file = os.path.join(misclassified_dir, "misclassified_log.json")
    with open(misclassified_log_file, "w") as f:
        json.dump(misclassified_info, f, indent=4)
    print(f"Misclassified image info saved to {misclassified_log_file}")

    # Save prediction probability information for each image
    probs_log_file = os.path.join(misclassified_dir, "all_image_probs.json")
    with open(probs_log_file, "w") as f:
        json.dump(all_image_info, f, indent=4)
    print(f"All image prediction probabilities saved to {probs_log_file}")

    # Write test log file
    with open(log_file, "w") as file:
        json.dump(log_data, file, indent=4)
    print(f"Test results saved to {log_file}")


if __name__ == '__main__':
    main()
