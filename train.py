import copy
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.functional import softmax, log_softmax, kl_div

from sklearn.model_selection import train_test_split

import config
from dataset import LabeledDataset, UnlabeledFolderDataset, get_labeled_image_folder
from utils import (
    aug_transform, org_transform,
    get_base_model, load_checkpoint, save_checkpoint, set_seed
)


def train(
    model: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    criterion,
    device: str = None
) -> float:
    """
    Train.
    Return loss.
    """
    model.train()
    model = model.to(device)

    total_loss = 0
    samples_cnt = 1e-9

    for i, (_, augmented_inputs, targets) in enumerate(dataloader):
        augmented_inputs = augmented_inputs.to(device)
        targets = targets.to(device)
        outputs = model(augmented_inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        samples_cnt += len(augmented_inputs)
        total_loss += loss.item() * len(augmented_inputs)
        print(f"[Training {i + 1}/{len(dataloader)}] Loss: {total_loss / samples_cnt:.4f}", end='\r')
    print()
    return total_loss / samples_cnt


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: str = None
) -> float:
    """
    Validate.
    Return loss.
    """
    model.eval()
    model = model.to(device)

    total_loss = 0
    samples_cnt = 1e-9
    correct_cnt = 0

    with torch.no_grad():
        for i, (inputs, _, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            pred = torch.argmax(outputs, dim=1)

            correct_cnt += torch.sum(pred == targets).cpu().item()

            samples_cnt += len(inputs)
            total_loss += loss.item() * len(inputs)
            print(f"[Training {i + 1}/{len(dataloader)}] Loss: {total_loss / samples_cnt:.4f} | Acc: {correct_cnt / samples_cnt:.4f}", end='\r')
    print()
    return total_loss / samples_cnt


def train_MPL(
    teacher: nn.Module,
    student: nn.Module,
    teacher_optimizer: Optimizer,
    student_optimizer: Optimizer,
    l_loader: DataLoader,
    u_loader: DataLoader,
    device: str = None
) -> float:
    """
    Training using Meta Pseudo Labels.
    Return the sum of student's loss and teacher's loss.
    """
    teacher.train()
    student.train()
    teacher = teacher.to(device)
    student = student.to(device)

    total_teacher_loss = 0
    total_student_loss = 0
    teacher_samples_cnt = 1e-9
    student_samples_cnt = 1e-9

    unlabeled_loader = iter(u_loader)

    for i, (labeled_inputs, augmented_labeled_inputs, labeled_targets) in enumerate(l_loader):
        # ==== 1. Sample labeled and unlabeled data ====
        try:
            unlabeled_inputs, augmented_unlabeled_inputs = next(unlabeled_loader)
        except StopIteration:
            unlabeled_loader = iter(u_loader)
            unlabeled_inputs, augmented_unlabeled_inputs = next(unlabeled_loader)


        labeled_inputs = labeled_inputs.to(device)
        augmented_labeled_inputs = augmented_labeled_inputs.to(device)
        labeled_targets = labeled_targets.to(device)
        unlabeled_inputs = unlabeled_inputs.to(device)
        augmented_unlabeled_inputs = augmented_unlabeled_inputs.to(device)

        # ==== 2. Generate pseudo labels ====
        pseudo_logits = teacher(augmented_unlabeled_inputs)
        pseudo_labels = pseudo_logits.argmax(dim=1).detach()  # hard pseudo-labels
        # Filter out low confidence samples
        mask = pseudo_logits[torch.arange(pseudo_logits.size(0)), pseudo_labels] > 0.5
        pseudo_labels = pseudo_labels[mask]
        augmented_unlabeled_inputs = augmented_unlabeled_inputs[mask]

        # ==== 3. Clone student and optimizer for simulated update ====
        student_clone = copy.deepcopy(student)
        optimizer_clone = AdamW(student_clone.parameters())
        optimizer_clone.load_state_dict(student_optimizer.state_dict())

        # ==== 4. Simulate student update on unlabeled data ====
        if augmented_unlabeled_inputs.size(0) > 1: # Need more than 1 sample, or the model's batch norm will raise error
            student_clone.train()
            unsup_outputs = student_clone(augmented_unlabeled_inputs)
            unsup_loss = criterion(unsup_outputs, pseudo_labels) 

            optimizer_clone.zero_grad()
            unsup_loss.backward()
            optimizer_clone.step()

        # ==== 5. Compute supervised loss on labeled data using cloned student ====
        student_clone.eval()
        outputs = student_clone(labeled_inputs)
        meta_loss  = criterion(outputs, labeled_targets)

        # ==== 6. teacher_sup_loss
        outputs = teacher(labeled_inputs)
        teacher_sup_loss = criterion(outputs, labeled_targets)
        
        # ==== 7. Backprop supervised loss through pseudo labels to teacher ====
        log_augmented_p = log_softmax(pseudo_logits, dim=-1)
        org_p       = softmax(teacher(unlabeled_inputs), dim=-1)
        uda_loss = kl_div(log_augmented_p, org_p, reduction="batchmean")

        teacher_optimizer.zero_grad()
        teacher_loss = uda_loss + meta_loss  + teacher_sup_loss
        # print(f"uda_loss {uda_loss} + meta_loss  {meta_loss } = teacher_loss {teacher_loss}")
        teacher_loss.backward()
        teacher_optimizer.step()

        # ==== 8. Update the real student model ====
        # Mix labeled and pseudo-labeled data
        student.train()
        all_inputs = torch.cat([augmented_labeled_inputs, augmented_unlabeled_inputs], dim=0)
        all_targets = torch.cat([labeled_targets, pseudo_labels], dim=0)

        student_outputs = student(all_inputs)
        student_loss = criterion(student_outputs, all_targets)

        student_optimizer.zero_grad()
        student_loss.backward()
        student_optimizer.step()

        # Print info
        n_labeled = len(augmented_labeled_inputs)
        n_total = n_labeled + len(augmented_unlabeled_inputs)

        teacher_samples_cnt += n_labeled
        total_teacher_loss += teacher_loss.item() * n_labeled

        student_samples_cnt += n_total
        total_student_loss += student_loss.item() * n_total

        avg_student_loss = total_student_loss / student_samples_cnt
        avg_teacher_loss = total_teacher_loss / teacher_samples_cnt

        print(f"[Training {i + 1}/{len(l_loader)}] Teacher Loss: {avg_teacher_loss:.4f} | Student Loss: {avg_student_loss:.4f} | uda {uda_loss:.4f}, meta {meta_loss :.4f}, tsl {teacher_sup_loss:.4f}" + " "*20, end='\r')
    print()
    return total_teacher_loss / teacher_samples_cnt + total_student_loss / student_samples_cnt



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    set_seed(config.SEED)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    labeled_image_paths, labels = get_labeled_image_folder(config.labeled_dataset_dir)

    l_dataset = LabeledDataset(image_paths=labeled_image_paths, labels=labels, transform=org_transform, aug_transform=aug_transform)
    u_dataset = UnlabeledFolderDataset(root_dir=config.unlabeled_dataset_dir, transform=org_transform, aug_transform=aug_transform)

    l_batch_size = 512
    u_batch_size = len(u_dataset)

    generator = torch.Generator().manual_seed(config.SEED)
    l_loader  = DataLoader(l_dataset, batch_size=l_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    u_loader  = DataLoader(u_dataset, batch_size=u_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    teacher_chkpt = "teacher.pt"
    student_chkpt = "student.pt"
    teacher = get_base_model()
    student = get_base_model()
    teacher = teacher.to(device)

    initial_lr = 2e-5
    weight_decay = 1e-4
    teacher_optimizer = AdamW(teacher.parameters(), lr=initial_lr/2, weight_decay=weight_decay)
    student_optimizer = AdamW(student.parameters(), lr=initial_lr, weight_decay=weight_decay)
    teacher = teacher.to(device)
    student = student.to(device)
    criterion = nn.CrossEntropyLoss()

    # ============================================
    print(f"\nTrain both the teacher and the student")

    min_loss = 1e9
    epochs = 1000
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        loss = train_MPL(teacher, student, teacher_optimizer, student_optimizer, l_loader, u_loader, device)
        if loss < min_loss:
            min_loss = loss
            save_checkpoint(teacher_chkpt, teacher, teacher_optimizer, epoch, loss)
            save_checkpoint(student_chkpt, student, student_optimizer, epoch, loss)


    # ============================================
    print(f"\nFine-tune student on labeled dataset")
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        labeled_image_paths, 
        labels, 
        test_size=0.1, 
        shuffle=True, 
        random_state=config.SEED, 
        stratify=labels
    )

    train_set = LabeledDataset(train_paths, train_labels, transform=org_transform, aug_transform=aug_transform)
    val_set   = LabeledDataset(test_paths, test_labels, transform=org_transform, aug_transform=aug_transform)

    train_loader = DataLoader(train_set, batch_size=l_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=l_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    [student, _, _, _] = load_checkpoint(student_chkpt, student)
    student_optimizer = AdamW(student.parameters(), lr=1e-6, weight_decay=weight_decay)

    epochs = 50
    min_loss = 1e9
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train(student, student_optimizer, train_loader, criterion, device)
        loss = validate(student, val_loader, criterion, device)
        if loss < min_loss:
            min_loss = loss
            save_checkpoint(config.model_checkpoint, student, student_optimizer, epoch, loss)
    