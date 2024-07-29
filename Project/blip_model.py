import requests
import PIL
from transformers import BlipProcessor, BlipForQuestionAnswering

import pandas as pd
import numpy as np

import torch
import random

from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from datasets import load_dataset
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import confusion_matrix
from evals import *

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)


root_path = '/home/smanduru/CS682Project/data/CUB_200_2011'

images_path = root_path + '/images'
attributes_path = root_path + '/attributes/'

images = []

with open(root_path + "/images.txt", "r") as file:
    
    for line in file:
        folder, image = line.strip().split('/')
        index, folder = folder.split(" ")
        images.append(images_path + '/' + folder + '/' + image)
        
data = pd.read_csv(attributes_path + "image_attribute_labels.txt", 
                   names=['img_index', 'attribute_index', 'attribute_value', 'certainity', 'unknown'],
                   delimiter=' ', on_bad_lines='warn')

ranges = [(153, 167), (10, 24), (198, 212)] ## Forehead, Wing, Belly, Leg(264, 278)
# ranges = [(136, 149), (294, 308), (249, 263)] ## Eye, Crown, Primary

df = data[data['attribute_value'] == 1]
df = df[df['attribute_index'].isin(sum([list(range(start, end + 1)) for start, end in ranges], []))]

df = df[['img_index', 'attribute_index', 'attribute_value']].reset_index(drop=True)


attributes = {}
with open('/home/smanduru/CS682Project/data/attributes.txt' , "r") as file:
    
    for line in file:
        
        attribute_index, attribute = line.split()
        attribute_qsn, attribute_value = attribute.split('::')
        attribute_qsn = attribute_qsn.split('_', 1)[-1]
        attribute_qsn = attribute_qsn.replace('_', ' ')
        attributes[attribute_index] = [int(attribute_index), attribute_qsn, attribute_value] 

att_df = pd.DataFrame.from_dict(attributes, orient='index', 
                                columns=['attribute_index', 'attribute_qsn', 'attribute_answer'])

merged_df = df.merge(att_df, how='inner', on='attribute_index')
merged_df = merged_df.sort_values(by=['img_index', 'attribute_index']).reset_index(drop = True)

merged_df = merged_df.iloc[:100]

bboxes = []
with open('/home/smanduru/CS682Project/data/CUB_200_2011/bounding_boxes.txt' , "r") as file:
    for line in file:
        if line:
            _, bbox = line.strip().split(" ", 1)
            x, y, w, h = map(int, map(float, bbox.split()))
            bbox = [x, y, w, h]
            bboxes.append(bbox)

# len(bboxes)


answer_candidates = ["blue", "brown", "iridescent", "purple",
                     "rufous", "grey", "yellow", "olive",
                     "green", "pink", "orange", "black",
                     "white", "red", "buff"]


# class VQADataset(torch.utils.data.Dataset):
#     """Caltech Bird Dataset"""

#     def __init__(self, dataset, processor, images):
#         self.dataset = dataset
#         self.processor = processor
#         self.images = images

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         # get image + text
#         question_part = self.dataset[idx]['attribute_qsn']
#         question = f"What is the {question_part} of the bird?"
#         answer = self.dataset[idx]['attribute_answer']
        
#         img_index = self.dataset[idx]['img_index']
#         image_path = images[int(img_index) - 1]
#         image = PIL.Image.open(image_path).convert("RGB")
        
#         text = question
#         encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
#         labels = self.processor.tokenizer.encode(
#             answer, max_length= 8, pad_to_max_length=True, return_tensors='pt'
#         )
#         encoding["labels"] = labels
#         # remove batch dimension
#         for k,v in encoding.items():  encoding[k] = v.squeeze()
#         return encoding

target_shape = (224, 224, 3)
class VQADataset(torch.utils.data.Dataset):
    """Caltech Bird Dataset"""

    def __init__(self, dataset, processor, images):
        self.dataset = dataset
        self.processor = processor
        self.images = images

    def __len__(self):
        return len(self.dataset)
    
    def preprocess_image(image_path, bounding_box, target_shape):
        x, y, w, h = map(int, bounding_box)
        
        cropped_image = image[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, target_shape[:2])  # Resize to target shape
        normalized_image = resized_image / 255.0  # Normalize pixel values
        return normalized_image

    def __getitem__(self, idx):
        # get image + text
        question_part = self.dataset[idx]['attribute_qsn']
        question = f"What is the {question_part} of the bird?"
        answer = self.dataset[idx]['attribute_answer']
        
        img_index = self.dataset[idx]['img_index']
        image_path = images[int(img_index) - 1]
        x, y, w, h = bboxes[int(img_index) - 1]
        image = PIL.Image.open(image_path).convert("RGB")
        cropped_image = image.crop((x, y, x + w, y + h))  # Crop image based on bounding box
        resized_image = cropped_image.resize(target_shape[:2], PIL.Image.Resampling.LANCZOS)
        image = np.array(resized_image) / 255.0
        
        text = question
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length= 8, pad_to_max_length=True, return_tensors='pt'
        )
        encoding["labels"] = labels
        # remove batch dimension
        for k,v in encoding.items():  encoding[k] = v.squeeze()
        return encoding

dataset = Dataset.from_pandas(merged_df).train_test_split(test_size=0.2)

# training_dataset = load_dataset("json", data_files="Data/train.jsonl", split="train[:90%]")
# valid_dataset = load_dataset("json", data_files="Data/train.jsonl", split="train[90%:]")
# print("Training sets: {} - Validating set: {}".format(len(training_dataset), len(valid_dataset)))


train_dataset = VQADataset(dataset=dataset['train'],
                          processor=processor, images = images)
valid_dataset = VQADataset(dataset=dataset['test'],
                          processor=processor, images = images)

batch_size = 12
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# total_correct = total_samples = 0
# for batch in tqdm(valid_dataloader, desc='Validating'):
#     inputs = batch.to(device)
#     labels = batch['labels'].to(device)
#     outputs = model.generate(**inputs)

#     actuals = [processor.tokenizer.decode(label, skip_special_tokens=True) for label in labels]
#     predicted = [processor.decode(out, skip_special_tokens=True) for out in outputs]
#     total_correct += sum(1 for pred, actual in zip(predicted, actuals) if pred == actual)
#     total_samples += labels.size(0)

# accuracy = total_correct / total_samples
# print('Accuracy on validation set:', accuracy)

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    target_modules=["query", "value"], #if you know the 
    lora_dropout=0.05,
    bias="none", # set this for CLM or Seq2Seq
)

model = get_peft_model(model, config)
model.print_trainable_parameters()


optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay = 0.05)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
num_epochs = 1
patience = 15
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()


for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for idx, batch in zip(tqdm(range(len(train_dataloader)), desc='Training batch: ...'), train_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_masked,
                        labels=labels)
            
        loss = outputs.loss
        epoch_loss += loss.item()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    model.eval()
    eval_loss = 0
    for idx, batch in zip(tqdm(range(len(valid_dataloader)), desc='Validating batch: ...'), valid_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_masked,
                        labels=labels)

        loss = outputs.loss
        eval_loss += loss.item()
    
    tracking_information.append((epoch_loss/len(train_dataloader), 
                                 eval_loss/len(valid_dataloader), 
                                 optimizer.param_groups[0]["lr"]))
    print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch+1,
                                                                          epoch_loss/len(train_dataloader),
                                                                          eval_loss/len(valid_dataloader),
                                                                          optimizer.param_groups[0]["lr"]))
    scheduler.step()
    if eval_loss < min_eval_loss:
        model.save_pretrained("Model/ep10_blip-saved-model", from_pt=True) 
        print("Saved model to Model/ep10_blip-saved-model")
        min_eval_loss = eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            break

pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
print("The finetuning process has done!")

## Set the model to evaluation mode

model.eval()  # Set the model to evaluation mode

total_correct = 0
total_samples = 0

all_actuals = []
all_predicted = []
all_wups = []
all_lds = []

with torch.no_grad():
    for batch in tqdm(valid_dataloader, desc='Validating'):
        inputs = batch.to(device)
        labels = batch['labels'].to(device)
        outputs = model.generate(**inputs)

        predicted = processor.batch_decode(outputs, skip_special_tokens=True)
        actuals = processor.batch_decode(labels, skip_special_tokens=True)
        _wups = [get_wups(ans, act, 0.9) for ans, act in zip(predicted, actuals)]
        _ld = [levenshtein_distance(ans, act) for ans, act in zip(predicted, actuals)]
        
        total_correct += sum(1 for pred, actual in zip(predicted, actuals) if pred == actual)
        total_samples += labels.size(0)
        
        all_actuals.extend(actuals)
        all_predicted.extend(predicted)
        all_wups.extend(_wups)
        all_lds.extend(_ld)

# Compute confusion matrix
conf_matrix = confusion_matrix(all_actuals, all_predicted)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)
print("WUPS and Levenshtein Distance: ", sum(all_wups)/len(all_wups), sum(all_lds)/len(all_lds))

# Calculate overall accuracy
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print('Accuracy on validation set:', accuracy, total_correct/total_samples)

bleu = evaluate.load("bleu")
results = bleu.compute(predictions=all_predicted, references=all_actuals)
print('BLEU', results['bleu'])

# Accuracy on validation set: 0.369780108936857
# Accuracy on validation set: 0.37088965099858784
# Accuracy on validation set: 0.43473875327819245 on finetuning bboxes

