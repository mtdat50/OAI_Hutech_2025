import os
import torch
from torch.utils.data import DataLoader

import config
from dataset import UnlabeledFolderDataset
from utils import (
    aug_transform, org_transform,
    get_base_model, load_checkpoint, set_seed, export_csv
)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    set_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("\nPredict on unlabeled dataset")
    predictions = []
    val_u_dataset   = UnlabeledFolderDataset(root_dir=config.unlabeled_dataset_dir, transform=org_transform, aug_transform=aug_transform)
    val_u_loader    = DataLoader(val_u_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)


    student = get_base_model()
    [student, _, _, _] = load_checkpoint(config.model_checkpoint, student)
    student.eval()
    student = student.to(device)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(val_u_loader):
            inputs = inputs.to(device)
            outputs = student(inputs)
            prediction = torch.argmax(outputs.data, 1).cpu().tolist()
            predictions.extend(prediction)
            print(f"Predicting {i + 1}/{len(val_u_loader)}...\r", end='')
        print("Predicting complete." + " " * 20)
    
    image_paths = val_u_dataset.get_all_image_paths()
    image_names = [os.path.basename(path).split(".")[0]  for path in image_paths]
    export_csv(config.submission_csv_path, predictions, image_names)