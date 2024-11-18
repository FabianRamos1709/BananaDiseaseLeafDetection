from transformers import ViTForImageClassification, Trainer, TrainingArguments, ViTFeatureExtractor
import wandb
import os
from datasets import ClassLabel, Dataset, Features, Image, Value, DatasetDict
from PIL import Image as PImage
import numpy as np
import torch
import gc


api_key = "e8cd58a9044abf8685565b9666fd1369865a602e"
wandb.login(key=api_key)

PROJECT = "DeteccionHojaBanano"
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
DATASET = "BananaLD"
wandb.init(project=PROJECT,  
           tags=[MODEL_NAME, DATASET],
           notes="Classification Fine Tune BananaLD dataset with google/vit-base-patch16-224-in21k") 

path_data_train = "C:/Users/sandi/OneDrive/Documentos/Universidad_Surcolombiana/SEPTIMO_SEMESTRE/ELECTIVA_COMPT/dataset_final/train"
path_data_validation = "C:/Users/sandi/OneDrive/Documentos/Universidad_Surcolombiana/SEPTIMO_SEMESTRE/ELECTIVA_COMPT/dataset_final/val"
path_data_test = "C:/Users/sandi/OneDrive/Documentos/Universidad_Surcolombiana/SEPTIMO_SEMESTRE/ELECTIVA_COMPT/dataset_final/test"


label_dic = {'BLACK_SIGATOKA': 0, 'HEALTHY': 1, 'MAL_DE_PANAMA': 2}

Clabels = ClassLabel(num_classes=3, names=[
                     "BLACK_SIGATOKA", "HEALTHY", "MAL_DE_PANAMA"])


list_images = []
list_path = []
list_labels = []
list_cats = []

for path, subdirs, files in os.walk(path_data_train):
    for name in files:
        if name.endswith('.jpg'): 
            abs_path = os.path.join(path, name)
            category = abs_path.replace("\\", "/").split("/")[-2]
            try:
                with PImage.open(abs_path) as image:
                    list_images.append(image.copy())
            except Exception as e:
                print(f"Error al abrir la imagen {abs_path}: {e}")
                continue 

            list_path.append(abs_path)
            list_labels.append(label_dic.get(category))
            list_cats.append(category)


dtrain = {
    "image": list_images,
    "image_file_path": list_path,
    "labels": list_labels
}
ds_train = Dataset.from_dict(mapping=dtrain, features=Features({
    "labels": Value(dtype='int32'),
    'image_file_path': Value(dtype='string'),
    'image': Image()
}))
ds_train = ds_train.cast_column("labels", Clabels)
ds_train = ds_train.shuffle(seed=42)

print(ds_train)

list_images = []
list_path = []
list_labels = []
list_cats = []

for path, subdirs, files in os.walk(path_data_validation):
    for name in files:
        if name.endswith('.jpg'):  
            abs_path = os.path.join(path, name)
            category = abs_path.replace("\\", "/").split("/")[-2]
            try:
                with PImage.open(abs_path) as image:
                    list_images.append(image.copy())
            except Exception as e:
                print(f"Error al abrir la imagen {abs_path}: {e}")
                continue  

            list_path.append(abs_path)
            list_labels.append(label_dic.get(category))
            list_cats.append(category)

dvalidation = {
    "image": list_images,
    "image_file_path": list_path,
    "labels": list_labels
}
ds_validation = Dataset.from_dict(mapping=dvalidation, features=Features({
    "labels": Value(dtype='int8'),
    'image_file_path': Value(dtype='string'),
    'image': Image()
}))
ds_validation = ds_validation.cast_column("labels", Clabels)

print(ds_validation)


list_images = []
list_path = []
list_labels = []
list_cats = []

for path, subdirs, files in os.walk(path_data_test):
    for name in files:
        if name.endswith('.jpg'):  
            abs_path = os.path.join(path, name)
            category = abs_path.replace("\\", "/").split("/")[-2]
            try:
                with PImage.open(abs_path) as image:
                    list_images.append(image.copy())
            except Exception as e:
                print(f"Error al abrir la imagen {abs_path}: {e}")
                continue  

            list_path.append(abs_path)
            list_labels.append(label_dic.get(category))
            list_cats.append(category)

dtest = {
    "image": list_images,
    "image_file_path": list_path,
    "labels": list_labels
}
ds_test = Dataset.from_dict(mapping=dtest, features=Features({
    "labels": Value(dtype='int8'),
    'image_file_path': Value(dtype='string'),
    'image': Image()
}))
ds_test = ds_test.cast_column("labels", Clabels)

print(ds_test)

ds = DatasetDict(
    {"train": ds_train, "validation": ds_validation, "test": ds_test})

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs


def transform(example_batch):
    print("Ejemplo batch recibido:", example_batch)  

    if 'image' not in example_batch or 'labels' not in example_batch:
        raise KeyError("Faltan claves en example_batch")

    images = example_batch['image']
    labels = example_batch['labels']

    if not isinstance(images, list):
        images = [images]

    inputs = feature_extractor(images, return_tensors='pt')
    
    inputs['labels'] = labels

    return inputs

prepared_ds = ds.with_transform(transform)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


metric = load_metric("accuracy",trust_remote_code=True)
def compute_metrics(p):

  return  metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

labels = ds['train'].features['labels'].names
id2label={str(i): c for i, c in enumerate(labels)}
label2id={c: str(i) for i, c in enumerate(labels)}


model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)

gc.collect()
torch.cuda.empty_cache()

training_args = TrainingArguments(
  output_dir="./vit-base-BananaLD-224",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=20,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=1e-4,
  weight_decay=1e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='wandb', 
  load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=feature_extractor,
)

with wandb.init(project=PROJECT, job_type="train"):
  train_results = trainer.train()
  trainer.save_model()
  trainer.log_metrics("train", train_results.metrics)
  trainer.save_metrics("train", train_results.metrics)
  trainer.save_state()
  print(train_results.metrics)
  
with wandb.init(project=PROJECT, job_type="evaluation"):
  metrics = trainer.evaluate(prepared_ds['test'])
  trainer.log_metrics("eval", metrics)
  trainer.save_metrics("eval", metrics)

print(trainer.optimizer)