import os
import torch
import random
import numpy as np
import torch.nn as nn

from torchvision.transforms import v2
from torch.optim.optimizer import Optimizer
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

from typing import Tuple, List


org_transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((32, 32)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
aug_transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomResizedCrop(32, scale=(0.8, 1.0)),
    v2.RandomPerspective(distortion_scale=0.2),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(20),
    v2.Resize((32, 32)),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_base_model(n_classes=4):
    model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    return model


def save_checkpoint(
    path: str, 
    model: nn.Module, 
    optimizer: Optimizer, 
    epoch: int, 
    loss: float
) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer,
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, path)    
    print(f"Saved checkpoint with epoch {epoch} and loss {loss:.4f}")


def load_checkpoint(
    path: str, 
    model: nn.Module
) -> Tuple[nn.Module, Optimizer, int, float]:
    """
    Returns:
        model.

        optimizer.

        epoch.

        loss.
    """
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = checkpoint["optimizer"]
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Loaded model with epoch {epoch} and loss {loss:.4f}")

    return [model, optimizer, epoch, loss]


def export_csv(export_file_path: str, result: List[int], image_names: List[str]) -> None:
    """
    Export to csv file with 2 columns format: idx, label.

    Args:
        export_file_path: Path to store the csv file.

        result: List of predicted labels.
    """
    dir = os.path.dirname(export_file_path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)

    with open(export_file_path, "w") as file:
        file.write("id,type\n")
        for i, res in enumerate(result):
            file.write(f"{image_names[i]},{res}\n")
    print(f"Exported results to {export_file_path}.")
