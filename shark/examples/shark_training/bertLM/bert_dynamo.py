# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys
from typing import List

from PIL import Image
import requests

import torch
import torch._dynamo as dynamo
import torchvision.models as models
from torchvision import transforms

import torch_mlir
from torch_mlir.dynamo import make_simple_dynamo_backend
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

import torch
from torch.nn.utils import _stateless
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shark.shark_trainer import SharkTrainer


class MiniLMSequenceClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/MiniLM-L12-H384-uncased",  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


def predictions(torch_func, jit_func, inp):
    golden_prediction = torch_func(inp)
    print("PyTorch prediction")
    print(golden_prediction)
    prediction = torch.from_numpy(jit_func(inp.numpy()))
    print("torch-mlir prediction")
    print(prediction)

@make_simple_dynamo_backend
def refbackend_torchdynamo_backend(fx_graph: torch.fx.GraphModule,
                                   example_inputs: List[torch.Tensor]):
    mlir_module = torch_mlir.compile(
        fx_graph, example_inputs, output_type="linalg-on-tensors")
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(mlir_module)
    loaded = backend.load(compiled)

    def compiled_callable(*inputs):
        inputs = [x.numpy() for x in inputs]
        result = loaded.forward(*inputs)
        if not isinstance(result, tuple):
            result = torch.from_numpy(result)
        else:
            result = tuple(torch.from_numpy(x) for x in result)
        return result
    return compiled_callable

def forward(params, buffers, args):
    params_and_buffers = {**params, **buffers}
    _stateless.functional_call(
        mod, params_and_buffers, args, {}
    ).sum().backward()
    optim = torch.optim.SGD(get_sorted_params(params), lr=0.01)
    # optim.load_state_dict(optim_state)
    optim.step()
    return params, buffers

bert = MiniLMSequenceClassification()
bert.train()
dynamo_callable = dynamo.optimize(refbackend_torchdynamo_backend)(bert)
inp = torch.randint(2, (1, 128))
packed_inputs = (
    dict(bert.named_parameters()),
    dict(bert.named_buffers()),
    tuple(inp),
)
predictions(bert.forward, lambda x: dynamo_callable(torch.from_numpy(x)).detach().numpy(), inp)
