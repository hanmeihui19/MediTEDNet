import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, margin=0.6, lambda_weight=0.2, alpha=1.5, gamma=0.8):
        """
        Args:
            num_classes (int): Number of classes
            feature_dim (int): Feature dimension
            margin (float): Margin used in the contrastive loss
            lambda_weight (float): Weighting factor between classification loss and contrastive loss
            alpha (float): Weighting factor for positive samples
            gamma (float): Weighting factor for negative samples

        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.lambda_weight = lambda_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, features, labels):
        """
        Args:
            features (Tensor): (batch_size, feature_dim)
            labels (Tensor): (batch_size,)

        Returns:
            loss (Tensor): contrastive_loss
        """
        # Get unique classes in the batch
        unique_labels = labels.unique()

        # Dynamically compute class prototypes
        dynamic_prototypes = []
        for label in unique_labels:
            mask = (labels == label).float().unsqueeze(1)  # (batch_size, 1)
            class_features = mask * features  # Retain features of the current class only
            class_sum = class_features.sum(dim=0)  # Sum of features
            class_count = mask.sum()  # Number of samples in this class
            prototype = class_sum / (class_count + 1e-8)  # Compute class prototype
            dynamic_prototypes.append(prototype)

        dynamic_prototypes = torch.stack(dynamic_prototypes)  # (num_classes_in_batch, feature_dim)

        # Compute contrastive loss
        contrastive_loss = 0
        for i, feature in enumerate(features):
            label = labels[i]

            # Positive prototype of the current sample
            pos_idx = (unique_labels == label).nonzero(as_tuple=True)[0].item()
            positive_prototype = dynamic_prototypes[pos_idx]

            # Negative prototype (choose the nearest negative class)
            negative_distances = torch.norm(dynamic_prototypes - feature.unsqueeze(0), dim=1)  # Distance to all prototypes
            negative_distances[pos_idx] = float('inf')  # Exclude the positive class
            neg_idx = negative_distances.argmin().item()  # Index of the nearest negative class
            negative_prototypes = dynamic_prototypes[neg_idx]  # Nearest negative prototype

            # Normalize distances
            d_ap = torch.norm(feature - positive_prototype, p=2) ** 2
            d_an = torch.norm(feature - negative_prototypes, p=2) ** 2
            d_ap = d_ap / (d_ap + d_an + 1e-8)
            d_an = d_an / (d_ap + d_an + 1e-8)

            # Compute contrastive loss (with weighted positive distance)
            contrastive_loss += torch.maximum(torch.tensor(0.0), self.alpha * d_ap - self.gamma * d_an + self.margin)

        contrastive_loss /= features.size(0)

        return torch.abs(self.lambda_weight * contrastive_loss)


