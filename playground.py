from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('./lms/bert-base-uncased')
model = BertForMaskedLM.from_pretrained('./lms/bert-base-uncased')

mask_token = tokenizer.mask_token

inputs = tokenizer(f"The capital of France is {mask_token}.", return_tensors='pt')

labels = tokenizer(f"Paris", add_special_tokens=False, return_tensors='pt')['input_ids']

print(f"torch version: {torch.__version__}")
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

outputs = model(**inputs, labels=labels)

logits = outputs.logits  # tensor(1, 9, 30522)
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]  # tensor(1)
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)  # tensor(1)
out_token = tokenizer.decode(predicted_token_id)

print(f"out token: {out_token}")
print(outputs)