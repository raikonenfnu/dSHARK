# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# import sys
# from typing import List

# from PIL import Image
# import requests

# import torch
# import torch._dynamo as dynamo
# import torchvision.models as models
# from torchvision import transforms
import numpy as np
import torch_mlir
from model import BaseModel
# from torch_mlir.dynamo import make_simple_dynamo_backend
# from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

import torch
import torch.nn as nn
import torch._dynamo as dynamo
from transformers import RobertaModel
import copy
from typing import List

from shark_backend import shark_torchdynamo_backend

# TODO: fix out_dim 1 during walrus codegen verifier due to map constraint.
# TODO: fix out_dim 1 during walrus codegen verifier due to map constraint.
inp_dim = 128
out_dim = 8
batch_size = 128
torch.manual_seed(0)
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
base_model = RobertaModel.from_pretrained("roberta-base")
# base_model = BertForPreTraining.from_pretrained('bert-large-uncased')
# medium BERT size L12_A12_H768. Large BERT L24_A16_H1024 causes OOM on GPU V100
my_config = copy.deepcopy(base_model.config)
my_config.num_hidden_layers = 2
my_config.num_attention_heads = 4
my_config.hidden_size = 16
my_config.vocab_size = 8192
my_config.attention_probs_dropout_prob=0.0
my_config.hidden_dropout_prob=0.0
net = RobertaModel(my_config)

input_ids = torch.randint(0, 5000, (8,128), dtype=torch.int32)

def train_func(input_ids):
    # result = net.forward(inp)
    outputs = net(input_ids=input_ids)
    loss = outputs.last_hidden_state
    return loss

dynamo_callable = dynamo.optimize(shark_torchdynamo_backend)(train_func)
torch.manual_seed(0)
res = dynamo_callable(input_ids)
print("res",res)

torch.manual_seed(0)
ref = train_func(input_ids)
print("ref", ref)