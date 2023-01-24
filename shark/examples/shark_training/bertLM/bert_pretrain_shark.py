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

# import torch_mlir
# from torch_mlir.dynamo import make_simple_dynamo_backend
# from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

import torch
from torch.nn.utils import _stateless
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForPreTraining
import copy

from shark.shark_trainer import SharkTrainer


def get_sorted_params(named_params):
    return [i[1] for i in sorted(named_params.items())]

class BertPretrainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base_model = BertForPreTraining.from_pretrained('bert-large-uncased')
        my_config = copy.deepcopy(base_model.config)
        my_config.num_hidden_layers = 1
        my_config.num_attention_heads = 2
        my_config.hidden_size = 16
        self.model = BertForPreTraining(my_config)

    def forward(self, input_ids, input_mask, segment_ids, labels, next_sentence_label):
        return self.model.forward(input_ids=input_ids,
                        attention_mask=input_mask,
                        token_type_ids=segment_ids,
                        labels=masked_lm_labels,
                        next_sentence_label=next_sentence_labels)


bert = BertPretrainer()
sample_input = torch.load("save_cpu_input.pt")
input_ids = sample_input["input_ids"].to("cpu")
input_mask = sample_input["input_mask"].to("cpu")
segment_ids = sample_input["segment_ids"].to("cpu")
masked_lm_labels = sample_input["masked_lm_labels"].to("cpu")
next_sentence_labels = sample_input["next_sentence_labels"].to("cpu")
packed_inputs = (input_ids,
                 input_mask,
                 segment_ids,
                 masked_lm_labels,
                 next_sentence_labels)
output = bert(*packed_inputs)
print(output)


def forward(params, buffers, packed_inputs):
    params_and_buffers = {**params, **buffers}
    _stateless.functional_call(
        bert, params_and_buffers, packed_inputs, {}
    ).loss.backward()
    optim = torch.optim.SGD(get_sorted_params(params), lr=0.01)
    optim.step()
    return params, buffers

shark_module = SharkTrainer(bert, packed_inputs)
shark_module.compile(forward)
shark_module.train(num_iters=10)
print("done training")

# # input_ids = ([8, 128]) torch.int64 [0,300000]
# # input_mask = ([8, 128]) torch.in64 [0,1]
# # segment_ids = ([8, 128]) torch.in64 [0,1]
# # masked_lm_labels = ([8, 128]) torch.int64 [-100 or [0,30000]]
# # next_sentence_labe = [8] torch.int74 [0,1]
# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
