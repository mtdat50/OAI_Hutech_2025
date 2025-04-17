# OAI_Hutech_2025

Olympiad in Artificial Intelligence at Ho Chi Minh 2025. 

The task is image classification.

Base model: **Efficienetv2_l** [[1]](#1)

Training method: **Meta Pseudo Labels**.
> "We present Meta Pseudo Labels, a semi-supervised learning method that achieves a new state-of-the-art top-1 accuracy of 90.2% on ImageNet, which is 1.6% better than the existing state-of-the-art. Like Pseudo Labels, Meta Pseudo Labels has a teacher network to generate pseudo labels on unlabeled data to teach a student network. However, unlike Pseudo Labels where the teacher is fixed, the teacher in Meta Pseudo Labels is constantly adapted by the feedback of the student's performance on the labeled dataset. As a result, the teacher generates better pseudo labels to teach the student." [[2]](#2)

## Set up enviroment
Create virtual environment:
```Shell
python3 -m venv .venv
```

Activate virtual environmnet:
```Shell
source .venv/bin/activate
which python3
```

Install dependencies:
```Shell
pip3 install -r requirements.txt
```

## Config paths
Modify variables in `config.py` to change paths.

Path to dataset folder:
```Python
data_dir = "./archive/aio-hutech"
```

Path to save model in training and load model in predicting:
```Python
model_checkpoint = "./models_chkpt/finetuned_student.pt"
```

Path to export predicted result:
```Python
submission_csv_path = "./submission.csv"
```

## Train model
Our method use both the labeled (train) and unlabeled (test) datasets, so please make sure the unlabeled set is also available for the trainning process.

```Shell
python3 train.py
```


## Predict on test dataset
```Shell
python3 predict.py
```

## References
<a id="1">[1]</a> Mingxing Tan and Quoc V. Le (2021).
EfficientNetV2: Smaller Models and Faster Training.
https://arxiv.org/abs/2104.0029


<a id="2">[2]</a> Hieu Pham and Zihang Dai and Qizhe Xie and Minh-Thang Luong and Quoc V. Le (2021).
Meta Pseudo Labels.
https://arxiv.org/abs/2003.10580