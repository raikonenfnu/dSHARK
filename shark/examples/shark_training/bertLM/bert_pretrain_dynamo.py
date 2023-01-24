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

import torch_mlir
# from torch_mlir.dynamo import make_simple_dynamo_backend
# from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

import torch
from torch.nn.utils import _stateless
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForPreTraining
import copy
from typing import List

from shark.shark_trainer import SharkTrainer


def get_sorted_params(named_params):
    return [i[1] for i in sorted(named_params.items())]

def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
    removed_indexes = []
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, (list, tuple)):
                node_arg = list(node_arg)
                node_args_len = len(node_arg)
                for i in range(node_args_len):
                    curr_index = node_args_len - (i + 1)
                    if node_arg[curr_index] is None:
                        removed_indexes.append(curr_index)
                        node_arg.pop(curr_index)
                node.args = (tuple(node_arg),)
                break

    if len(removed_indexes) > 0:
        fx_g.graph.lint()
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
    removed_indexes.sort()
    return removed_indexes


def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                return len(node_arg) == 0
    return False

def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
    """
    Replace tuple with tuple element in functions that return one-element tuples.
    Returns true if an unwrapping took place, and false otherwise.
    """
    unwrapped_tuple = False
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    unwrapped_tuple = True
                    break

    if unwrapped_tuple:
        fx_g.graph.lint()
        fx_g.recompile()
    return unwrapped_tuple

from torch_mlir.dynamo import make_simple_dynamo_backend
import torch._dynamo as dynamo
torch._dynamo.config.verbose = True

@make_simple_dynamo_backend
def shark_torchdynamo_backend(
    fx_graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    if _returns_nothing(fx_graph):
        return fx_graph
    removed_none_indexes = _remove_nones(fx_graph)
    was_unwrapped = _unwrap_single_tuple_return(fx_graph)
    print(fx_graph.code)
    mlir_module = torch_mlir.compile(
        fx_graph, example_inputs, output_type="linalg-on-tensors"
    )

    from shark.shark_inference import SharkInference
    import io

    bytecode_stream = io.BytesIO()
    mlir_module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    shark_module = SharkInference(
        mlir_module=bytecode, device="cpu", mlir_dialect="tm_tensor"
    )
    shark_module.compile()

    def compiled_callable(*inputs):
        inputs = [x.numpy() for x in inputs]
        result = shark_module("forward", inputs)
        if was_unwrapped:
            result = [
                result,
            ]
        if not isinstance(result, list):
            result = torch.from_numpy(result)
        else:
            result = tuple(torch.from_numpy(x) for x in result)
            result = list(result)
            for removed_index in removed_none_indexes:
                result.insert(removed_index, None)
            result = tuple(result)
        return result

    return compiled_callable

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

# Model setup.
bert = BertPretrainer()

# Sample input setup.
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

#  Optimizer setup.
param_optimizer = list(bert.named_parameters())
no_decay = ['bias', 'LayerNorm'] # gamma/beta are in LayerNorm.weight

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
lr = 4e-4
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr)
optimizer.zero_grad()

# Setup compile training function.
def train_func(packed_inputs):
    loss = bert.forward(*packed_inputs).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss

dynamo_callable = dynamo.optimize(shark_torchdynamo_backend)(train_func)
for i in range(10):
    print("loss:",dynamo_callable(packed_inputs))

# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
