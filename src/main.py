import random
import os

import numpy as np
from sklearn.metrics import classification_report, f1_score

os.environ['TRANSFORMERS_CACHE'] = '../.cache'

from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

import torch
import data

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2ForSequenceClassification.from_pretrained("stanford-crfm/BioMedLM").to(device)
for param in model.transformer.parameters():
    param.requires_grad = False
model.train()
model.config.pad_token_id = 28895
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1.6e-4)

train, test = data.get_datasets()

train_labels = [1] * len(train[0]) + [0] * len(train[0])
inputs = train[0] + train[1]
inputs = [f"Does {entity1} increase the risk of {entity2}?" for entity1, entity2 in inputs]
inputs = list(zip(inputs, train_labels))
random.shuffle(inputs)

batch_size = 32
epochs = 20
for e in range(epochs):
    total_loss = 0
    for i in range(len(inputs) // batch_size + 1):
        optimizer.zero_grad()
        batch = inputs[i * batch_size:(i + 1) * batch_size]
        input_ids = tokenizer(
            [x[0] for x in batch], return_tensors="pt", padding=True
        ).to(device)

        sample_output = model(input_ids["input_ids"], attention_mask=input_ids['attention_mask'], labels=torch.tensor([
            x[1] for x in batch]).long().to(device))
        loss = sample_output["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss

    print(total_loss / (i + 1))
    torch.save(model.state_dict(), f"model_{e + 1}.pkl")


model.eval()

test_labels = [1] * len(test[0]) + [0] * len(test[0])
inputs = test[0] + test[1]
inputs = [f"Does {entity1} increase the risk of {entity2}?" for entity1, entity2 in inputs]
preds = []
for i in range(len(inputs) // batch_size + 1):
    optimizer.zero_grad()
    batch = inputs[i * batch_size:(i + 1) * batch_size]
    input_ids = tokenizer(batch, return_tensors="pt", padding=True
    ).to(device)

    sample_output = model(input_ids["input_ids"], attention_mask=input_ids['attention_mask'])
    logits = sample_output["logits"].cpu().detach().numpy()
    batch_preds = np.argmax(logits, axis=1).tolist()
    preds.extend(batch_preds)

print(classification_report(test_labels, preds, target_names=["Negative", "Positive"]))
