import os
import torch
from PIL import Image
from torch.utils.data import Dataset

from typing import List, Tuple


custom_class_to_idx = {
    'nấm mỡ': 0, 
    'bào ngư xám + trắng': 1, 
    'Đùi gà Baby (cắt ngắn)': 2, 
    'linh chi trắng': 3
}

class LabeledDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform = None, aug_transform = None):
        super().__init__()

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.aug_transform = aug_transform


    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        transformed_image = None
        augmented_image = None

        if self.transform:
            transformed_image = self.transform(image)
        
        # Apply augmentations
        if self.aug_transform:
            augmented_image = self.aug_transform(image)

        return transformed_image, augmented_image, self.labels[idx]
    

class UnlabeledFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform = None, aug_transform = None):
        super().__init__()

        self.image_paths = []
        self.transform = transform
        self.aug_transform = aug_transform

        for file_name in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file_name)

            # Check if file is image
            try:
                Image.open(file_path)
            except:
                continue
            
            self.image_paths.append(file_path)


    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            torch.Tensor: tensor image.

            torch.Tensor: tensor of augmented image.
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")

        transformed_image = None
        augmented_image = None

        if self.transform:
            transformed_image = self.transform(image)
        
        # Apply augmentations
        if self.aug_transform:
            augmented_image = self.aug_transform(image)

        return transformed_image, augmented_image
    
    
    def get_all_image_paths(self):
        return self.image_paths.copy()
    

def get_labeled_image_folder(root_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load image paths and their labels from root_dir.
    Path format: root_dir/label/image_name

    Returns:
        image_paths(List[str]): Paths to images.

        labels(List[int]): Labels of images.
    """
    image_paths = []
    labels = []

    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        # print(dir_path)

        if not os.path.isdir(dir_path) or dir_name not in custom_class_to_idx:
            continue
        
        label = custom_class_to_idx[dir_name]
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)

            # Check if file is image
            try:
                Image.open(file_path)
            except:
                continue
            
            image_paths.append(file_path)
            labels.append(label)

    return image_paths, labels
