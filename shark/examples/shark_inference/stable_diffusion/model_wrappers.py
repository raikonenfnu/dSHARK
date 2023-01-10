from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
from utils import compile_through_fx
from stable_args import args
import torch

model_config = {
    "v2_1": "stabilityai/stable-diffusion-2-1",
    "v2_1base": "stabilityai/stable-diffusion-2-1-base",
    "v1_4": "CompVis/stable-diffusion-v1-4",
}

# clip has 2 variants of max length 77 or 64.
model_clip_max_length = 64 if args.max_length == 64 else 77
if args.variant in ["anythingv3", "analogdiffusion", "dreamlike"]:
    model_clip_max_length = 77
elif args.variant == "openjourney":
    model_clip_max_length = 64

model_variant = {
    "stablediffusion": "SD",
    "anythingv3": "Linaqruf/anything-v3.0",
    "dreamlike": "dreamlike-art/dreamlike-diffusion-1.0",
    "openjourney": "prompthero/openjourney",
    "analogdiffusion": "wavymulder/Analog-Diffusion",
}

model_input = {
    "v2_1": {
        "clip": (torch.randint(1, 2, (2, model_clip_max_length)),),
        "vae": (torch.randn(1, 4, 96, 96),),
        "unet": (
            torch.randn(1, 4, 96, 96),  # latents
            torch.tensor([1]).to(torch.float32),  # timestep
            torch.randn(2, model_clip_max_length, 1024),  # embedding
            torch.tensor(1).to(torch.float32),  # guidance_scale
        ),
    },
    "v2_1base": {
        "clip": (torch.randint(1, 2, (2, model_clip_max_length)),),
        "vae": (torch.randn(1, 4, 64, 64),),
        "unet": (
            torch.randn(1, 4, 64, 64),  # latents
            torch.tensor([1]).to(torch.float32),  # timestep
            torch.randn(2, model_clip_max_length, 1024),  # embedding
            torch.tensor(1).to(torch.float32),  # guidance_scale
        ),
    },
    "v1_4": {
        "clip": (torch.randint(1, 2, (2, model_clip_max_length)),),
        "vae": (torch.randn(1, 4, 64, 64),),
        "unet": (
            torch.randn(1, 4, 64, 64),
            torch.tensor([1]).to(torch.float32),  # timestep
            torch.randn(2, model_clip_max_length, 768),
            torch.tensor(1).to(torch.float32),
        ),
    },
}

# revision param for from_pretrained defaults to "main" => fp32
model_revision = {
    "stablediffusion": "main",
    "anythingv3": "diffusers",
    "analogdiffusion": "main",
    "openjourney": "main",
    "dreamlike": "main",
}


def get_clip_mlir(model_name="clip_text", extra_args=[]):

    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    if args.variant == "stablediffusion":
        if args.version != "v1_4":
            text_encoder = CLIPTextModel.from_pretrained(
                model_config[args.version], subfolder="text_encoder"
            )

    elif args.variant in [
        "anythingv3",
        "analogdiffusion",
        "openjourney",
        "dreamlike",
    ]:
        text_encoder = CLIPTextModel.from_pretrained(
            model_variant[args.variant],
            subfolder="text_encoder",
            revision=model_revision[args.variant],
        )
    else:
        raise ValueError(f"{args.variant} not yet added")

    class CLIPText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder

        def forward(self, input):
            return self.text_encoder(input)[0]

    clip_model = CLIPText()
    shark_clip = compile_through_fx(
        clip_model,
        model_input[args.version]["clip"],
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_clip


def get_base_vae_mlir(model_name="vae", extra_args=[]):
    class BaseVaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                model_config[args.version]
                if args.variant == "stablediffusion"
                else model_variant[args.variant],
                subfolder="vae",
                revision=model_revision[args.variant],
            )

        def forward(self, input):
            x = self.vae.decode(input, return_dict=False)[0]
            return (x / 2 + 0.5).clamp(0, 1)

    vae = BaseVaeModel()
    use_fp16 = False
    if args.variant == "stablediffusion":
        if args.precision == "fp16":
            use_fp16 = True
        #     vae = vae.half().cuda()
        #     inputs = tuple(
        #         [
        #             inputs.half().cuda()
        #             for inputs in model_input[args.version]["vae"]
        #         ]
        #     )
        # else:
            inputs = model_input[args.version]["vae"]
    elif args.variant in [
        "anythingv3",
        "analogdiffusion",
        "openjourney",
        "dreamlike",
    ]:
        inputs = model_input["v1_4"]["vae"]
        if args.precision == "fp16":
            use_fp16 = True
        #     vae = vae.half().cuda()
        #     inputs = tuple(
        #         [inputs.half().cuda() for inputs in model_input["v1_4"]["vae"]]
        #     )
        # else:
        #     inputs = model_input["v1_4"]["vae"]
    else:
        raise ValueError(f"{args.variant} not yet added")

    shark_vae = compile_through_fx(
        vae,
        inputs,
        model_name=model_name,
        extra_args=extra_args,
        use_fp16=use_fp16
    )
    return shark_vae


def get_vae_mlir(model_name="vae", extra_args=[]):
    class VaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                model_config[args.version]
                if args.variant == "stablediffusion"
                else model_variant[args.variant],
                subfolder="vae",
                revision=model_revision[args.variant],
            )

        def forward(self, input):
            input = 1 / 0.18215 * input
            x = self.vae.decode(input, return_dict=False)[0]
            x = (x / 2 + 0.5).clamp(0, 1)
            x = x * 255.0
            return x.round()

    vae = VaeModel()
    use_fp16 = False
    if args.variant == "stablediffusion":
        if args.precision == "fp16":
            use_fp16 = True
        #     vae = vae.half().cuda()
        #     inputs = tuple(
        #         [
        #             inputs.half().cuda()
        #             for inputs in model_input[args.version]["vae"]
        #         ]
        #     )
        # else:
        inputs = model_input[args.version]["vae"]
    elif args.variant in [
        "anythingv3",
        "analogdiffusion",
        "openjourney",
        "dreamlike",
    ]:
        if args.precision == "fp16":
            use_fp16 = True
        #     vae = vae.half().cuda()
        #     inputs = tuple(
        #         [inputs.half().cuda() for inputs in model_input["v1_4"]["vae"]]
        #     )
        # else:
        inputs = model_input["v1_4"]["vae"]
    else:
        raise ValueError(f"{args.variant} not yet added")

    shark_vae = compile_through_fx(
        vae,
        inputs,
        model_name=model_name,
        extra_args=extra_args,
        use_fp16=use_fp16
    )
    return shark_vae


def get_unet_mlir(model_name="unet", extra_args=[]):
    class UnetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = UNet2DConditionModel.from_pretrained(
                model_config[args.version]
                if args.variant == "stablediffusion"
                else model_variant[args.variant],
                subfolder="unet",
                revision=model_revision[args.variant],
            )
            self.in_channels = self.unet.in_channels
            self.train(False)

        def forward(self, latent, timestep, text_embedding, guidance_scale):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latents = torch.cat([latent] * 2)
            unet_out = self.unet.forward(
                latents, timestep, text_embedding, return_dict=False
            )[0]
            noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            return noise_pred

    unet = UnetModel()
    use_fp16 = False
    if args.variant == "stablediffusion":
        if args.precision == "fp16":
            use_fp16 = True
        #     unet = unet.half().cuda()
        #     inputs = tuple(
        #         [
        #             inputs.half().cuda() if len(inputs.shape) != 0 else inputs
        #             for inputs in model_input[args.version]["unet"]
        #         ]
        #     )
        # else:
        inputs = model_input[args.version]["unet"]
    elif args.variant in [
        "anythingv3",
        "analogdiffusion",
        "openjourney",
        "dreamlike",
    ]:
        if args.precision == "fp16":
            use_fp16 = True
        #     unet = unet.half().cuda()
        #     inputs = tuple(
        #         [
        #             inputs.half().cuda() if len(inputs.shape) != 0 else inputs
        #             for inputs in model_input["v1_4"]["unet"]
        #         ]
        #     )
        # else:
        inputs = model_input["v1_4"]["unet"]
    else:
        raise ValueError(f"{args.variant} is not yet added")
    shark_unet = compile_through_fx(
        unet,
        inputs,
        model_name=model_name,
        extra_args=extra_args,
        use_fp16 = use_fp16
    )
    return shark_unet
