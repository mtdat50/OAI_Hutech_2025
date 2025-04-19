### START: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###
SEED = 0  # Số seed (Ban tổ chức sẽ công bố & thay đổi vào lúc chấm)
# Đường dẫn đến thư mục train
# (đúng theo cấu trúc gồm 4 thư mục cho 4 classes của ban tổ chức)
TRAIN_DATA_DIR_PATH = 'data/train'
# Đường dẫn đến thư mục test
TEST_DATA_DIR_PATH = 'data/test'
### END: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###

### START: CÁC THƯ VIỆN IMPORT ###
# Lưu ý: các thư viện & phiên bản cài đặt vui lòng để trong requirements.txt
import os
import random
import numpy as np
import torch
import torch.nn as nn
# import tensorflow as tf

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

### END: CÁC THƯ VIỆN IMPORT ###

### START: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###
# Seeding nhằm đảm bảo kết quả sẽ cố định
# và không ngẫu nhiên ở các lần chạy khác nhau
# Set seed for random
random.seed(SEED)
# Set seed for numpy
np.random.seed(SEED)
# Set seed for torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Set seed for tensorflow
# tf.random.set_seed(SEED)
### END: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###

# START: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###
from libs.train import train, validate, train_MPL
from libs.dataset import LabeledDataset, UnlabeledFolderDataset, get_labeled_image_folder
from libs.utils import (
    aug_transform, org_transform,
    get_base_model, load_checkpoint, save_checkpoint, export_csv
)
### END: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###


### START: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###
# Model sẽ được train bằng cac ảnh ở [TRAIN_DATA_DIR_PATH]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

labeled_image_paths, labels = get_labeled_image_folder(TRAIN_DATA_DIR_PATH)

l_dataset = LabeledDataset(image_paths=labeled_image_paths, labels=labels, transform=org_transform, aug_transform=aug_transform)
u_dataset = UnlabeledFolderDataset(root_dir=TEST_DATA_DIR_PATH, transform=org_transform, aug_transform=aug_transform)

l_batch_size = 512
u_batch_size = 256

l_loader  = DataLoader(l_dataset, batch_size=l_batch_size, shuffle=True, num_workers=4, pin_memory=True)
u_loader  = DataLoader(u_dataset, batch_size=u_batch_size, shuffle=True, num_workers=4, pin_memory=True)

teacher_chkpt = "teacher.pt"
student_chkpt = "student.pt"
teacher = get_base_model()
student = get_base_model()
teacher = teacher.to(device)

initial_lr = 1e-3
weight_decay = 1e-4
teacher_optimizer = AdamW(teacher.parameters(), lr=initial_lr/2, weight_decay=weight_decay)
student_optimizer = AdamW(student.parameters(), lr=initial_lr, weight_decay=weight_decay)
teacher_lr = ReduceLROnPlateau(teacher_optimizer, factor=0.2, patience=20)
student_lr = ReduceLROnPlateau(student_optimizer, factor=0.2, patience=20)
teacher = teacher.to(device)
student = student.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ============================================
print(f"\nTrain both the teacher and the student")

min_loss = 1e9
epochs = 1000
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"LR {student_lr.get_last_lr()}")
    loss = train_MPL(teacher, student, teacher_optimizer, student_optimizer, criterion, l_loader, u_loader, device)

    teacher_lr.step(loss)
    student_lr.step(loss)

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
    random_state=SEED, 
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
model_checkpoint = "model_chkpt/finetuned_student.pt"
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train(student, student_optimizer, train_loader, criterion, device)
    loss = validate(student, val_loader, criterion, device)
    if loss < min_loss:
        min_loss = loss
        save_checkpoint(model_checkpoint, student, student_optimizer, epoch, loss)
    
### END: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###


### START: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###
# Kết quả dự đoán của mô hình cho tập dữ liệu các ảnh ở [TEST_DATA_DIR_PATH]
# sẽ lưu vào file "output/results.csv"
# Cấu trúc gồm 2 cột: image_name và label: (encoded: 0, 1, 2, 3)
# image_name,label
# image1.jpg,0
# image2.jpg,1
# image3.jpg,2
# image4.jpg,3

submission_csv_path = "output/results.csv"

print("\nPredict on unlabeled dataset")
predictions = []
val_u_dataset   = UnlabeledFolderDataset(root_dir=TEST_DATA_DIR_PATH, transform=org_transform, aug_transform=aug_transform)
val_u_loader    = DataLoader(val_u_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

[student, _, _, _] = load_checkpoint(model_checkpoint, student)
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
image_names = [os.path.basename(path)  for path in image_paths]
export_csv(submission_csv_path, predictions, image_names)

### END: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###