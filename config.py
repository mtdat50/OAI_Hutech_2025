import os
import kagglehub

SEED = 42

# data_dir = kagglehub.competition_download("aio-hutech")
data_dir = "archive/aio-hutech"
labeled_dataset_dir = os.path.join(data_dir, 'train')
unlabeled_dataset_dir = os.path.join(data_dir, 'test')

# model_checkpoint = kagglehub.model_download("tmaitn/oai_hutech_2025_finetuned_student.pt/pyTorch/default") # download trained model
model_checkpoint = "./models_chkpt/finetuned_student.pt"
submission_csv_path = "./submission.csv"