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
import numpy as np
from datetime import datetime

from utils.metrics import metrics_cal_multiclass
from utils.utils import create_lr_scheduler,EarlyStopping,seed

from loss.tripletLoss import TripletLoss

from Model import VSSM as MediTEDNet # import model

# set random seed
seed(0)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = datasets.ImageFolder(root="the path of your train set",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, worker_init_fn=np.random.seed(0))

    validate_dataset = datasets.ImageFolder(root="the path of your validation set",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, worker_init_fn=np.random.seed(0))
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    num_classes = len(train_dataset.classes)

    net = MediTEDNet(num_classes=num_classes)

    if net is None:
        raise ValueError("Model instance `net` is None.")

    net = net.to(device)
    print("Model successfully moved to device.")

    # Compute FLOPs
    def compute_flops(model, input_tensor):
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total()

    # Compute the number of model parameters
    def compute_params(model):
        return sum(p.numel() for p in model.parameters())

    flops = compute_flops(net, torch.randn(1, 3, 224, 224).to(device))
    params = compute_params(net)

    args = {
        'contrast_fea_size': 512,  # The projected feature dimension is 512
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }

    # Loss function
    loss_constractive_function = TripletLoss(num_classes, args['contrast_fea_size'], margin=0.6, lambda_weight=0.2, alpha=1.5, gamma=0.8, beta=0.0).to(args['device'])
    loss_classifier_function = nn.CrossEntropyLoss()

    epochs = 150

    best_acc = 0.0

    save_path = './{}Net.pth'.format(model_name)
    optimizer = optim.AdamW(net.parameters(), lr=0.0002
                            , weight_decay=1e-4)

    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs,
    #                                    warmup=True, warmup_epochs=10)

    # Initialize Early Stopping
    # early_stopping = EarlyStopping(patience=30,delta=1e-4)

    # Define the log file path
    log_file = "/{}Net.txt".format(model_name)
    with open(log_file, 'w') as f:
        f.write("Training Log\n")

    train_steps = len(train_loader)

    for epoch in range(epochs):
        for param_group in optimizer.param_groups:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch + 1}] Learning Rate: {current_lr:.6f}")

        # train
        net.train()

        train_loss_all = 0.0
        loss_constractive_all = 0.0
        loss_classifier_all = 0.0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data

            feature_outputs, classifier_outputs = net(images.to(device))
            loss_constractive = loss_constractive_function(feature_outputs, labels.to(device))
            loss_classifier = loss_classifier_function(classifier_outputs, labels.to(device))
            train_loss = loss_constractive + loss_classifier

            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()
            # update lr
            # lr_scheduler.step()

            loss_constractive_all += loss_constractive.item()
            loss_classifier_all += loss_classifier.item()
            train_loss_all += train_loss.item()

            train_bar.desc = "train epoch[{}/{}] loss_constractive:{:.3f} loss_classifier:{:.3f} loss:{:.3f}".format(epoch + 1,
                                                                     epochs,loss_constractive,loss_classifier,train_loss)

        # validate
        net.eval()

        val_loss_all = 0.0

        all_preds = []
        all_labels = []
        all_probs = []

        val_accuracy = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_outputs = net(val_images.to(device))
                val_loss = loss_classifier_function(val_outputs, val_labels.to(device))
                val_loss_all += val_loss.item()

                predict_y = torch.max(val_outputs, dim=1)[1]
                probs = torch.softmax(val_outputs, dim=1)
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                val_accuracy += torch.eq(predict_y, val_labels.to(device)).sum().item()

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        val_loss_all /= len(validate_loader)
        val_accurate = val_accuracy / val_num

        conf_matrix, precision, recall, f1, specificity, auc, per_class_metrics = metrics_cal_multiclass(all_labels, all_preds, all_probs)

        log_data = {
            'epoch': epoch + 1,
            'timestamp': timestamp,
            'train_constractive_loss': float(loss_constractive_all / train_steps),
            'train_classifier_loss': float(loss_classifier_all / train_steps),
            'train_loss': float(train_loss_all / train_steps),
            'val_loss': float(val_loss_all),
            'val_accurate': float(val_accurate),
            'val_precision': float(precision),
            'val_recall': float(recall),
            'val_f1': float(f1),
            'val_auc': float(auc),
            'val_specificity': float(specificity),
            'current_lr': float(current_lr),
            'FLOPs': float(flops / 1e9),
            'Params': float(params / 1e6),
            'conf_matrix': conf_matrix.tolist(),
            'per_class_metrics': per_class_metrics
        }

        # Print logs to the console
        print(f"[epoch {epoch + 1}] "
              f"train_constractive_loss: {log_data['train_constractive_loss']:.3f} "
              f"train_classifier_loss: {log_data['train_classifier_loss']:.3f} "
              f"train_loss: {log_data['train_loss']:.3f} "
              f"val_loss: {log_data['val_loss']:.3f} "
              f"val_accurate: {log_data['val_accurate']:.3f} "
              f"val_precision: {log_data['val_precision']:.3f} "
              f"val_recall: {log_data['val_recall']:.3f} "
              f"val_f1: {log_data['val_f1']:.3f} "
              f"val_auc: {log_data['val_auc']:.3f} "
              f"val_specificity: {log_data['val_specificity']:.3f} "
              f"FLOPs: {log_data['FLOPs']:.3f} GFLOPs "
              f"Params: {log_data['Params']:.3f} MParams "
              f"current_lr: {log_data['current_lr']:.6f}"
              )

        # **Write to the log file**
        try:
            # Read the log file (initialize as empty list if file is empty)
            with open(log_file, "r") as file:
                try:
                    logs = json.load(file)
                except json.JSONDecodeError:
                    logs = []  # File is empty
        except FileNotFoundError:
            logs = []  # File does not exist

        # Append the current epoch's log data
        logs.append(log_data)

        # Write all logs back to the file
        with open(log_file, "w") as file:
            json.dump(logs, file, indent=4)
            file.write("\n\n")

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print(f"Best model saved at {save_path} with accuracy {best_acc:.3f}")

        # # Check early stopping
        # early_stopping(val_loss_all, net, save_path)
        #
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    print('Finished Training')


if __name__ == '__main__':
    main()
