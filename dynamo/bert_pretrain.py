import numpy as np
import torch
import torch.nn as nn
import torch._dynamo as dynamo
from transformers import BertForPreTraining
import copy

from shark_backend import shark_torchdynamo_backend

inp_dim = 128
out_dim = 8
batch_size = 128
torch.manual_seed(0)
base_model = BertForPreTraining.from_pretrained('bert-large-uncased')
base_model = base_model.train()
# medium BERT size L12_A12_H768. Large BERT L24_A16_H1024 causes OOM on GPU V100
my_config = copy.deepcopy(base_model.config)
my_config.num_hidden_layers = 1
my_config.num_attention_heads = 1
my_config.hidden_size = 16
my_config.vocab_size = 8192
net = BertForPreTraining(my_config)

input_ids = torch.randint(0, 5000, (8,128), dtype=torch.int64)
input_mask = torch.randint(0, 2, (8,128), dtype=torch.int64)
input_mask = torch.ones([8,128], dtype=torch.int64)
masked_lm_labels = torch.randint(0, 3000, (8,128), dtype=torch.int64)
next_sentence_labels = torch.randint(0, 2, (8,), dtype=torch.int64)
segment_ids = torch.randint(0, 2, (8,128), dtype=torch.int64)


torch.set_grad_enabled(True)
net.train()
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)
def train_func(input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels):
    loss = (net(input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids, labels=masked_lm_labels, next_sentence_label=next_sentence_labels).loss)
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    return loss

torch.manual_seed(0)
print("compiling")
dynamo_callable = dynamo.optimize(shark_torchdynamo_backend)(train_func)
# dynamo_callable = dynamo.optimize("inductor")(train_func)
print("running")
res = dynamo_callable(input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels)
print("res",res)

# res = dynamo_callable(input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels)
# print("res",res)

for i in range(100):
    res = dynamo_callable(input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels)
    print("res",res)

# torch.manual_seed(0)
# ref = train_func(input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels)
# print("ref", ref)

# ref = train_func(input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels)
# print("ref", ref)
