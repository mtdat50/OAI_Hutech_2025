import copy
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.functional import softmax, log_softmax, kl_div


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
            print(f"[Validating {i + 1}/{len(dataloader)}] Loss: {total_loss / samples_cnt:.4f} | Acc: {correct_cnt / samples_cnt:.4f}", end='\r')
    print()
    return total_loss / samples_cnt


def train_MPL(
    teacher: nn.Module,
    student: nn.Module,
    teacher_optimizer: Optimizer,
    student_optimizer: Optimizer,
    criterion,
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
        outputs = teacher(augmented_labeled_inputs)
        teacher_sup_loss = criterion(outputs, labeled_targets)
        
        # ==== 7. Backprop supervised loss through pseudo labels to teacher ====
        T = 0.8
        UDA_factor = 1
        log_augmented_p = log_softmax(pseudo_logits, dim=-1)
        org_p           = softmax(teacher(unlabeled_inputs) / T, dim=-1).detach()
        uda_loss        = kl_div(log_augmented_p, org_p, reduction="batchmean") * UDA_factor

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