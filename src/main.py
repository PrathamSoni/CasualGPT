from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

import torch
import data

device = torch.device("cpu")

tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")
model = GPT2ForSequenceClassification.from_pretrained("stanford-crfm/BioMedLM").to(device)

train, test = data.get_datasets()

train_labels = [1] * len(train[0]) + [0] * len(train[0])
inputs = train[0] + train[1]
inputs = [f"Does {entity1} increase the risk of {entity2}?" for entity1, entity2 in inputs]

batch_size = 8
input_ids = tokenizer.encode(
    [inputs[:batch_size]], return_tensors="pt"
).to(device)

sample_output = model(input_ids, labels = train_labels)

print("Output:\n" + 100 * "-")
print(sample_output)
