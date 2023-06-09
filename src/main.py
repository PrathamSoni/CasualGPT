import random

from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

import torch
import data

device = torch.device("cpu")

tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2ForSequenceClassification.from_pretrained("stanford-crfm/BioMedLM").to(device)
for param in model.transformer.parameters():
    param.requires_grad = False
model.train()
model.config.pad_token_id = 28895
optimizer = torch.optim.Adam(model.parameters(), lr = 1.6e-4)

train, test = data.get_datasets()

train_labels = [1] * len(train[0]) + [0] * len(train[0])
inputs = train[0] + train[1]
inputs = [f"Does {entity1} increase the risk of {entity2}?" for entity1, entity2 in inputs]
inputs = list(zip(inputs, train_labels))
random.shuffle(inputs)

batch_size = 64
for i in tqdm(range(len(inputs)//batch_size)):
    optimizer.zero_grad()
    batch = inputs[i * batch_size:(i + 1) * batch_size]
    input_ids = tokenizer(
        [x[0] for x in batch], return_tensors="pt", padding=True
    ).to(device)

    sample_output = model(input_ids['input_ids'], attention_mask = input_ids['attention_mask'], labels=torch.tensor([
        x[1] for x in batch]).long())
    loss = sample_output["loss"]
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pkl")