import os

os.environ["AMD_ENABLE_LLPC"] = "1"

from transformers import CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
import torchvision.transforms as T
from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from tqdm.auto import tqdm
import numpy as np
from random import randint
from stable_args import args

# This has to come before importing cache objects
if args.clear_all:
    print("CLEARING ALL, EXPECT SEVERAL MINUTES TO RECOMPILE")
    from glob import glob
    import shutil

    vmfbs = glob(os.path.join(os.getcwd(), "*.vmfb"))
    for vmfb in vmfbs:
        if os.path.exists(vmfb):
            os.remove(vmfb)
    home = os.path.expanduser("~")
    if os.name == "nt":  # Windows
        appdata = os.getenv("LOCALAPPDATA")
        shutil.rmtree(os.path.join(appdata, "AMD/VkCache"), ignore_errors=True)
        shutil.rmtree(os.path.join(home, "shark_tank"), ignore_errors=True)
    elif os.name == "unix":
        shutil.rmtree(os.path.join(home, ".cache/AMD/VkCache"))
        shutil.rmtree(os.path.join(home, ".local/shark_tank"))


from utils import set_init_device_flags

from opt_params import get_unet, get_vae, get_clip
from model_wrappers import get_unet_torch
from schedulers import (
    SharkEulerDiscreteScheduler,
)
import time
import sys
from shark.iree_utils.compile_utils import dump_isas

def gpu_transform_fx(fx_g):
    kwargs_dict = {
        "dtpye": torch.float32,
        "device": torch.device(type="cuda"),
        "pin_memory": False
    }
    for node in fx_g.graph.nodes:
        if node.kwargs["device"] == torch.device(type="cpu"):
            node.kwargs["device"] = torch.device(type="gpu")
        # if node.op == "call_function":
        #     if node.target in [torch.ops.aten.arange, torch.ops.aten.empty]:
        #         node.kwargs = kwargs_dict
    fx_g.graph.lint()

def transform_fx(fx_g):

    kwargs_dict = {
        "dtype": torch.float16,
        "device": torch.device(type="cuda"),
        "pin_memory": False,
    }
    for node in fx_g.graph.nodes:
        if node.op == "call_function":
            if node.target in [
                torch.ops.aten.arange,
                torch.ops.aten.empty
            ]:
                node.kwargs = kwargs_dict
            if node.target in [torch.ops.aten.var_mean]:
                with fx_g.graph.inserting_before(node):
                    new_node = fx_g.graph.call_function(
                        torch.ops.prims.convert_element_type,
                        args=(node.args[0], torch.float32),
                        kwargs={},
                    )
                    node.args = (new_node, node.args[1])
            if node.name.startswith("getitem"):
                with fx_g.graph.inserting_before(node):
                    if node.args[0].target in [torch.ops.aten.var_mean]:
                        new_node = fx_g.graph.call_function(
                            torch.ops.aten._to_copy,
                            args=(node,),
                            kwargs={"dtype": torch.float16},
                        )
                        node.append(new_node)
                        node.replace_all_uses_with(new_node)
                        new_node.args = (node,)
                        new_node.kwargs = {"dtype": torch.float16}
            # aten.empty should be filled with zeros.
            if node.target in [torch.ops.aten.empty]:
                with fx_g.graph.inserting_after(node):
                    new_node = fx_g.graph.call_function(
                        torch.ops.aten.zero_,
                        args=(node,),
                    )
                    node.append(new_node)
                    node.replace_all_uses_with(new_node)
                    new_node.args = (node,)
        # if node.op == "output":
        #     import pdb; pdb.set_trace()

    fx_g.graph.lint()

def compile_fx(
    model, inputs, is_f16=False, f16_input_mask=None, debug=False
):
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._decomp import get_decompositions

    # TODO: Control the decompositions.
    fx_g = make_fx(
        model,
        decomposition_table=get_decompositions(
            [
                torch.ops.aten.embedding_dense_backward,
                torch.ops.aten.native_layer_norm_backward,
                torch.ops.aten.slice_backward,
                torch.ops.aten.select_backward,
                torch.ops.aten.norm.ScalarOpt_dim,
                torch.ops.aten.native_group_norm,
                torch.ops.aten.upsample_bilinear2d.vec,
                torch.ops.aten.split.Tensor,
                torch.ops.aten.split_with_sizes,
                torch.ops.aten.native_layer_norm,
            ]
        ),
    )(*inputs)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    def strip_overloads(gm):
        """
        Modifies the target of graph nodes in :attr:`gm` to strip overloads.
        Args:
            gm(fx.GraphModule): The input Fx graph module to be modified
        """
        for node in gm.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                node.target = node.target.overloadpacket
        gm.recompile()

    strip_overloads(fx_g)

    if is_f16:
        fx_g = fx_g.half()
        transform_fx(fx_g)
        fx_g.recompile()

    if model._get_name() == "UnetModel":
        with open("unet_cpu.fx", "w") as text_file:
            text_file.write(str(fx_g.graph))

    ts_graph = torch.jit.script(fx_g)
    return ts_graph

if __name__ == "__main__":

    dtype = torch.float32 if args.precision == "fp32" else torch.half

    prompt = args.prompts
    neg_prompt = args.negative_prompts
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    if args.version == "v2_1":
        height = 768
        width = 768

    num_inference_steps = args.steps  # Number of denoising steps

    # Scale for classifier-free guidance
    guidance_scale = torch.tensor(args.guidance_scale).to(torch.float32)

    # Handle out of range seeds.
    uint32_info = np.iinfo(np.uint32)
    uint32_min, uint32_max = uint32_info.min, uint32_info.max
    seed = args.seed
    if seed < uint32_min or seed >= uint32_max:
        seed = randint(uint32_min, uint32_max)
    generator = torch.manual_seed(
        seed
    )  # Seed generator to create the inital latent noise

    # TODO: Add support for batch_size > 1.
    batch_size = len(prompt)
    if batch_size != 1:
        sys.exit("More than one prompt is not supported yet.")
    if batch_size != len(neg_prompt):
        sys.exit("prompts and negative prompts must be of same length")

    set_init_device_flags()
    if args.version == "v2_1":
        unet_input = np.load("unet_input.npz", allow_pickle = True)
    else:
        unet_input = np.load("unet_512_input.npz", allow_pickle = True)

    # unet = get_unet()
    unet = get_unet_torch()
    sample_input = [torch.from_numpy(unet_input["latent_model_input"]).float(), torch.from_numpy(unet_input["timestep"]).float(), torch.from_numpy(unet_input["text_embeddings_numpy"]).float(), torch.from_numpy(unet_input["guidance_scale"])]
    use_f16 = True
    unet = compile_fx(unet, sample_input, use_f16, [True, True, True, False]).cuda()
    with torch.no_grad():
            noise_pred = unet.forward(torch.from_numpy(unet_input["latent_model_input"]).cuda(), torch.from_numpy(unet_input["timestep"]).cuda(), torch.from_numpy(unet_input["text_embeddings_numpy"]).cuda(), torch.from_numpy(unet_input["guidance_scale"]).cuda())
    noise_pred.to("cpu")
    # del unet
    # torch.cuda.empty_cache()
    # unet_torch = get_unet_torch()
    # if args.precision == "fp16":
    #     unet_torch = unet_torch.cuda().half().cuda()
    # original_noise_pred = unet_torch.forward(torch.from_numpy(unet_input["latent_model_input"]).cuda(), torch.from_numpy(unet_input["timestep"]).cuda(), torch.from_numpy(unet_input["text_embeddings_numpy"]).cuda(), torch.from_numpy(unet_input["guidance_scale"]).cuda())
    # print(original_noise_pred)
    # noise_pred = unet(
    #     "forward",
    #     (
    #         unet_input["latent_model_input"],
    #         unet_input["timestep"],
    #         unet_input["text_embeddings_numpy"],
    #         unet_input["guidance_scale"],
    #     ),
    #     send_to_host=False,
    # )
    # noise_pred = noise_pred.to_host()
    # TODO: Validate that it's only faulty on v2_1 and v2_1 base still works.
    # TODO: Modify to output the bmm and it's args and compare with IREE results.
    print(noise_pred)

