import math
import numpy as np
import torch
import random
import os

import torch
import math


def create_lr_scheduler(optimizer, num_step: int, epochs: int,
                        warmup=True, warmup_epochs=10,
                        warmup_factor=1e-3, end_factor=1e-6):
    """
    Custom learning rate scheduler with Warmup and Cosine Annealing
    """
    assert num_step > 0 and epochs > 0

    if warmup is False:
        warmup_epochs = 0

    def lr_lambda(current_step: int):
        """
        Compute the learning rate factor for the current step
        """
        total_steps = epochs * num_step
        warmup_steps = warmup_epochs * num_step

        if current_step <= warmup_steps:
            # Warmup phase: linear increase
            alpha = float(current_step) / float(warmup_steps)
            return warmup_factor * (1 - alpha) + alpha
        else:
            # Cosine annealing phase
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return end_factor + 0.5 * (1 - end_factor) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # torch.use_deterministic_algorithms(True)
    print("setting random seed：", seed)


class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)