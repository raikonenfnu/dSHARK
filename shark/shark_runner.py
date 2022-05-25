# Copyright 2020 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch.utils._python_dispatch import enable_torch_dispatch_mode
from torch_mlir.eager_mode import torch_mlir_tensor
from torch_mlir.eager_mode.torch_mlir_tensor import TorchMLIRTensor
from torch_mlir_e2e_test.eager_backends.refbackend import EagerModeRefBackend

from shark.iree_eager_backend import EagerModeIREELinalgOnTensorsBackend
from shark.torch_mlir_utils import get_torch_mlir_module, export_module_to_mlir_file, run_on_refbackend
from shark.iree_utils import get_results, get_iree_compiled_module, export_iree_module_to_vmfb, build_benchmark_args, run_benchmark
import os
from shark.parser import shark_args
from tqdm import tqdm
import time


class SharkRunner:
    """Base class for Shark Inference and Shark Runner."""

    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool = False,
        device: str = None,
        jit_trace: bool = False,
        from_aot : bool = False,
        frontend : str = "torch",
        benchmark_mode :bool = False
    ):
        self.model = model
        self.frontend_model = model
        self.from_aot = from_aot
        self.input = input
        self.frontend = frontend
        self.benchmark_mode = benchmark_mode
        device = device if device is not None else shark_args.device
        if self.frontend in ["pytorch", "torch"]:
            self.model = get_torch_mlir_module(self.model, input, dynamic,
                                                        jit_trace,
                                                        from_aot)
        (
            self.iree_compilation_module,
            self.iree_config,
        ) = get_iree_compiled_module(self.model, device)

        # Debugging and Benchmark Options:
        if shark_args.save_mlir:
            export_module_to_mlir_file(self.model,
                                       shark_args.repro_dir)
        if shark_args.save_mlir or self.benchmark_mode:
            vmfb_file = export_iree_module_to_vmfb(self.model, device,
                            shark_args.repro_dir)
        if self.benchmark_mode:
            self.benchmark_cl = build_benchmark_args(vmfb_file, device, input, from_aot)

    # All the timings and benchmarking can be done here.
    def forward(self, input, frontend):
        return get_results(self.iree_compilation_module, input,
                           self.iree_config, frontend)

######### Benchmark Related Functions ###########

    def benchmark_mode(func):
        def inner(self, *args, **kwargs):
            assert self.benchmark_mode, "SharkRunner needs to be in benchmark mode to run benchmark methods."
            return func(self, *args, **kwargs)
        return inner

    @benchmark_mode
    def benchmark_frontend(self, inputs):
        if self.frontend in ["pytorch", "torch"]:
            self.benchmark_torch(inputs)
        elif self.frontend in ["tensorflow", "tf"]:
            self.benchmark_tf(inputs)

    @benchmark_mode
    def benchmark_torch(self, inputs):
        inputs = self.input if self.from_aot else inputs
        inputs = inputs[0]
        for i in range(shark_args.num_warmup_iterations):
            self.frontend_model.forward(inputs)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = self.frontend_model.forward(inputs)
            if i == shark_args.num_iterations - 1:
                end = time.time()
                break
        print(f"Torch benchmark:{shark_args.num_iterations/(end-begin)} iter/second, Total Iterations:{shark_args.num_iterations}")

    @benchmark_mode
    def benchmark_tf(self, inputs):
        print(f"TF benchmark not implemented yet!")
        return

    @benchmark_mode
    def benchmark_c(self):
        print(self.benchmark_cl)
        result = run_benchmark(self.benchmark_cl)
        print(f"Shark-C benchmark:{result} iter/second")

    @benchmark_mode
    def benchmark_python(self, inputs):
        inputs = self.input if self.from_aot else inputs
        input_list = [x.detach().numpy() for x in inputs]
        for i in range(shark_args.num_warmup_iterations):
            self.forward(input_list, self.frontend)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = self.forward(input_list, self.frontend)
            if i == shark_args.num_iterations - 1:
                end = time.time()
        print(f"Shark-Python benchmark:{shark_args.num_iterations/(end-begin)} iter/second, Total Iterations:{shark_args.num_iterations}")

    @benchmark_mode
    def benchmark_all(self, inputs):
        self.benchmark_frontend(inputs)
        self.benchmark_python(inputs)
        self.benchmark_c()

class SharkMode:

    def __init__(self, device="cpu"):
        if device == "refbackend":
            torch_mlir_tensor.backend = EagerModeRefBackend()
        else:
            torch_mlir_tensor.backend = EagerModeIREELinalgOnTensorsBackend(
                device)
        self.guard = enable_torch_dispatch_mode(TorchMLIRTensor)
        self.guard.__enter__()

    def __del__(self):
        self.guard.__exit__(None, None, None)
    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool = False,
        device: str = None,
        jit_trace: bool = False,
        from_aot: bool = False,
        custom_inference_fn=None,
    ):
        self.model = model
        self.input = input
        self.from_aot = from_aot

        self.device = device if device is not None else shark_args.device

        self.shark_runner = SharkRunner(self.model, self.input, dynamic,
                                        self.device, jit_trace, from_aot)

    def benchmark_forward(self, inputs):
        inputs = self.input if self.from_aot else inputs
        input_list = [x.detach().numpy() for x in inputs]
        for i in range(shark_args.num_warmup_iterations):
            self.shark_runner.forward(input_list)

        for i in range(shark_args.num_iterations):
            begin = time.time()
            out = self.shark_runner.forward(input_list)
            end = time.time()
            print("Iteration " + str(i) + ": " + str(end - begin))
            if i == shark_args.num_iterations - 1:
                return out

    def forward(self, inputs):
        # TODO Capture weights and inputs in case of AOT, Also rework the
        # forward pass.
        inputs = self.input if self.from_aot else inputs
        input_list = [x.detach().numpy() for x in inputs]
        return self.shark_runner.forward(input_list)

    def run_on_refbackend(self, inputs):
        self.shark_runner.run_on_refbackend(inputs)
