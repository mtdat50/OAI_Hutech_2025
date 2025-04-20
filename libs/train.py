import torch
import higher
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.functional import softmax, log_softmax, kl_div


def train(
    model: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    criterion,
    device: str = None,
    use_uda=False
) -> float:
    """
    Train.
    Return loss.
    """
    model.train()
    model = model.to(device)

    total_loss = 0
    samples_cnt = 1e-9

    for i, (inputs, augmented_inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        augmented_inputs = augmented_inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        sup_loss = criterion(outputs, targets)

        uda_loss = torch.tensor(0).to(device)
        if use_uda:
            T = 0.8
            UDA_factor = 1
            log_augmented_p = log_softmax(model(augmented_inputs), dim=-1)
            org_p           = softmax(outputs / T, dim=-1).detach()
            uda_loss        = kl_div(log_augmented_p, org_p, reduction="batchmean") * UDA_factor

        loss = sup_loss + uda_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        samples_cnt += len(augmented_inputs)
        total_loss += loss.item() * len(augmented_inputs)
        print(f"[Training {i + 1}/{len(dataloader)}] Loss: {total_loss / samples_cnt:.4f} | uda {uda_loss.item():.4f} | sup_loss {sup_loss.item():.4f}", end='\r')
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
    device: str = None,
    threshold: float = 0.8,
    uda_weight_max: float = 1.0,
    uda_rampup_steps: int = 5000
) -> float:
    """
    Args:
        threshold: confidence threshold for real student update.
        uda_weight_max: maximum UDA consistency weight (lambda_UDA).
        uda_rampup_steps: steps to linearly ramp UDA weight from 0→uda_weight_max.
    Returns:
        Combined avg teacher + student loss.
    """
    teacher.train()
    student.train()
    teacher.to(device); student.to(device)

    total_teacher_loss = 0.0
    total_student_loss = 0.0
    teacher_cnt = 1e-9
    student_cnt = 1e-9
    global_step = 0
    unlabeled_iter = iter(u_loader)

    for i, (x_l, x_l_aug, y_l) in enumerate(l_loader):
        # 1) fetch unlabeled
        try:
            x_u, x_u_aug = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(u_loader)
            x_u, x_u_aug = next(unlabeled_iter)

        # move to device
        x_l, x_l_aug, y_l = x_l.to(device), x_l_aug.to(device), y_l.to(device)
        x_u, x_u_aug       = x_u.to(device), x_u_aug.to(device)

        # 2) teacher pseudo-labels (soft and hard)
        with torch.no_grad():
            t_logits_u_aug = teacher(x_u_aug)
            soft_pseudo = softmax(t_logits_u_aug, dim=-1)
            hard_pseudo = soft_pseudo.argmax(dim=-1)

        # 3) ramp UDA weight
        global_step += 1
        uda_weight = uda_weight_max * min(1.0, global_step / uda_rampup_steps)

        # 4) inner-loop student update (differentiable)
        with higher.innerloop_ctx(
            student, 
            student_optimizer,
            copy_initial_weights=False,
            track_higher_grads=True
        ) as (student_fmodel, diffopt):
            # a) old supervised loss
            with torch.no_grad():
                sup_old = criterion(student_fmodel(x_l), y_l)

            # b) unsup loss on soft pseudo-labels
            unsup_out = student_fmodel(x_u_aug)
            unsup_loss = kl_div(log_softmax(unsup_out, dim=-1),
                                soft_pseudo, reduction='batchmean')

            diffopt.step(unsup_loss)

            # c) new supervised loss
            sup_new = criterion(student_fmodel(x_l), y_l)

        # 5) compute reward = drop-gradient(sup_old - sup_new)
        reward = (sup_old - sup_new).detach()

        # 6) teacher losses
        # 6a) supervised on labeled
        t_sup = criterion(teacher(x_l), y_l)
        # 6b) UDA consistency on unlabeled
        log_aug = log_softmax(t_logits_u_aug, dim=-1)
        org_logits_u = teacher(x_u)
        orig = softmax(org_logits_u, dim=-1)
        uda_loss = kl_div(log_aug, orig, reduction='batchmean')

        # 6c) MPL pseudo-labeled cross-entropy
        mpl_loss = criterion(t_logits_u_aug, hard_pseudo)

        teacher_optimizer.zero_grad()
        teacher_loss = (
            t_sup
            + uda_weight * uda_loss
            + mpl_loss * reward
        )
        teacher_loss.backward()
        teacher_optimizer.step()

        # 7) real student update on mixed pseudo-labeled data
        student.train()
        # mask high-confidence
        conf, _ = soft_pseudo.max(dim=1)
        mask = conf > threshold
        x_mix = torch.cat([x_l_aug, x_u_aug[mask]], dim=0)
        y_mix = torch.cat([y_l,   hard_pseudo[mask]], dim=0)

        s_out = student(x_mix)
        student_loss = criterion(s_out, y_mix)
        student_optimizer.zero_grad()
        student_loss.backward()
        student_optimizer.step()

        # 8) logging & averaging
        n_l = x_l_aug.size(0)
        n_tot = n_l + mask.sum().item()
        teacher_cnt += n_l
        total_teacher_loss += teacher_loss.item() * n_l
        student_cnt += n_tot
        total_student_loss += student_loss.item() * n_tot

        print(f"[{i+1}/{len(l_loader)}] Teacher: {total_teacher_loss/teacher_cnt:.4f} | Student: {total_student_loss/student_cnt:.4f}", end="\r")
    print()

    return (total_teacher_loss/teacher_cnt
          + total_student_loss/student_cnt)
