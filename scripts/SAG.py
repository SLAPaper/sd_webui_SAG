import math
import os
import sys
import typing as tg
from inspect import isfunction

import gradio as gr
import PIL
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from torch import einsum, nn

import modules.scripts as scripts
from modules import shared
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import (
    AfterCFGCallbackParams,
    CFGDenoisedParams,
    CFGDenoiserParams,
    on_cfg_after_cfg,
    on_cfg_denoised,
    on_cfg_denoiser,
)
from scripts import xyz_grid_support_sag

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class LoggedSelfAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = tg.cast(int, default(context_dim, query_dim))

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attn_probs = None

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type="cuda"):
                q, k = q.float(), k.float()
                sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(tg.cast(torch.Tensor, mask), "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        self.attn_probs = sim

        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


def xattn_forward_log(
    self,
    x,
    context=None,
    mask: torch.Tensor | None = None,
    additional_tokens=None,
    n_times_crossframe_attn_in_self=0,
):
    h = self.heads

    n_tokens_to_mask = 0
    if additional_tokens is not None:
        # get the number of masked tokens at the beginning of the output sequence
        n_tokens_to_mask = additional_tokens.shape[1]
        # add additional token
        x = torch.cat([additional_tokens, x], dim=1)

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    if n_times_crossframe_attn_in_self:
        # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
        assert x.shape[0] % n_times_crossframe_attn_in_self == 0
        n_cp = x.shape[0] // n_times_crossframe_attn_in_self
        k = repeat(k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)
        v = repeat(v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION == "fp32":
        with torch.autocast(enabled=False, device_type="cuda"):
            q, k = q.float(), k.float()
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
    else:
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

    del q, k

    if exists(mask):
        mask = rearrange(tg.cast(torch.Tensor, mask), "b ... -> b (...)")
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, "b j -> (b h) () j", h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    self.attn_probs = sim
    global current_selfattn_map
    current_selfattn_map = sim

    out = einsum("b i j, b j d -> b i d", sim, v)
    out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
    if additional_tokens is not None:
        # remove additional token
        out = out[:, n_tokens_to_mask:]
    out = self.to_out(out)
    global current_outsize
    current_outsize = out.shape[-2:]
    return out


saved_original_selfattn_forward = None
current_selfattn_map = None
current_sag_guidance_scale = 1.0
sag_enabled = False
sag_mask_threshold = 1.0
sag_mask_threshold_auto = False
last_sag_mask_thresholds: list[float] = []

current_xin = None
current_outsize = (64, 64)
current_batch_size = 1
current_degraded_pred = None
current_unet_kwargs = {}
current_uncond_pred = None
current_degraded_pred_compensation = None

last_attn_masks: list[torch.Tensor] = []

# blur setting
blur_kernel_size = 9
blur_sigma = 1.0


def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


class Script(scripts.Script):
    def __init__(self):
        pass

    def title(self):
        return "Self Attention Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def denoiser_callback(self, parms: CFGDenoiserParams):
        if not sag_enabled:
            return
        global current_xin, current_batch_size

        # logging current uncond size for cond/uncond output separation
        current_batch_size = parms.text_uncond.shape[0]
        # logging current input for eps calculation later
        current_xin = parms.x[-current_batch_size:]

        # logging necessary information for SAG pred
        current_uncond_emb = parms.text_uncond
        current_sigma = parms.sigma
        current_image_cond_in = parms.image_cond
        global current_unet_kwargs
        current_unet_kwargs = {
            "sigma": current_sigma[-current_batch_size:],
            "image_cond": current_image_cond_in[-current_batch_size:],
            "text_uncond": current_uncond_emb,
        }

    def denoised_callback(self, params: CFGDenoisedParams):
        if not sag_enabled:
            return
        # output from DiscreteEpsDDPMDenoiser is already pred_x0
        uncond_output: torch.Tensor = params.x[-current_batch_size:]
        original_latents = uncond_output
        global current_uncond_pred
        current_uncond_pred = uncond_output

        # Produce attention mask
        # We're only interested in the last current_batch_size*head_count slices of logged self-attention map
        if current_selfattn_map is None:
            print("SAG WARNING: no current_selfattn_map, return", file=sys.stderr)
            return
        attn_map = current_selfattn_map[-current_batch_size * 8 :]
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = 8

        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_map_gap = attn_map.mean(1, keepdim=False).sum(1, keepdim=False)
        mask_threshold = sag_mask_threshold
        if sag_mask_threshold_auto:
            mask_threshold = attn_map_gap.mean().item() * sag_mask_threshold
        attn_mask: torch.Tensor = attn_map_gap > mask_threshold

        # check if is SDXL with 1 less down sampling
        if attn_mask.numel() == b * math.ceil(latent_h / 8) * math.ceil(latent_w / 8):
            middle_layer_latent_size = [
                math.ceil(latent_h / 8),
                math.ceil(latent_w / 8),
            ]
        elif attn_mask.numel() == b * math.ceil(latent_h / 4) * math.ceil(latent_w / 4):
            middle_layer_latent_size = [
                math.ceil(latent_h / 4),
                math.ceil(latent_w / 4),
            ]
        else:
            print(
                "SAG WARNING: unknown attention shape",
                attn_mask.size(),
                ", return",
                file=sys.stderr,
            )
            return

        attn_mask = (
            attn_mask.reshape(
                b, middle_layer_latent_size[0], middle_layer_latent_size[1]
            )
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )

        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # Blur according to the self-attention mask
        degraded_latents = gaussian_blur_2d(
            original_latents, kernel_size=blur_kernel_size, sigma=blur_sigma
        )
        degraded_latents = degraded_latents * attn_mask + original_latents * (
            1 - attn_mask
        )

        renoised_degraded_latent = degraded_latents - (uncond_output - current_xin)
        # renoised_degraded_latent = degraded_latents
        # get predicted x0 in degraded direction
        global current_degraded_pred_compensation
        current_degraded_pred_compensation = uncond_output - degraded_latents
        if shared.sd_model.model.conditioning_key == "crossattn-adm":
            make_condition_dict = lambda c_crossattn, c_adm: {
                "c_crossattn": c_crossattn,
                "c_adm": c_adm,
            }
        else:
            if isinstance(current_unet_kwargs["text_uncond"], dict):
                make_condition_dict = lambda c_crossattn, c_concat: {
                    **c_crossattn[0],
                    "c_concat": [c_concat],
                }
            else:
                make_condition_dict = lambda c_crossattn, c_concat: {
                    "c_crossattn": c_crossattn,
                    "c_concat": [c_concat],
                }
        degraded_pred = params.inner_model(
            renoised_degraded_latent,
            current_unet_kwargs["sigma"],
            cond=make_condition_dict(
                [current_unet_kwargs["text_uncond"]],
                [current_unet_kwargs["image_cond"]],
            ),
        )
        global current_degraded_pred
        current_degraded_pred = degraded_pred
        global last_attn_masks
        last_attn_masks.append(attn_mask)
        global last_sag_mask_thresholds
        last_sag_mask_thresholds.append(mask_threshold)

    def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams):
        if not sag_enabled:
            return

        if current_degraded_pred is None:
            print("SAG WARNING: no current_degraded_pred, return", file=sys.stderr)
            return

        params.x = params.x + (
            current_uncond_pred
            - (current_degraded_pred + current_degraded_pred_compensation)
        ) * float(current_sag_guidance_scale)
        # params.output_altered = True

    def ui(self, is_img2img):
        with gr.Accordion("Self Attention Guidance", open=False):
            enabled = gr.Checkbox(label="Enabled", default=False)
            scale = gr.Slider(
                label="Scale", minimum=-2.0, maximum=10.0, step=0.01, value=0.75
            )
            auto_th = gr.Checkbox(value=False, label="Auto threshold")
            mask_threshold = gr.Slider(
                label="SAG Mask Threshold",
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                value=1.0,
            )
            with gr.Row():
                blur_size = gr.Slider(
                    label="Blur kernel size", minimum=1, maximum=33, step=2, value=9
                )
                blur_sigma = gr.Slider(
                    label="Blur sigma", minimum=0.0, maximum=32.0, step=0.01, value=1.0
                )
            show_simmap = gr.Checkbox(value=False, label="Show attention map.")

        self.infotext_fields = [
            (enabled, "SAG Enabled"),
            (scale, "SAG Guidance Scale"),
            (mask_threshold, "SAG Mask Threshold"),
            (auto_th, "SAG Auto Threshold"),
            (blur_size, "SAG Blur Kernel Size"),
            (blur_sigma, "SAG Blur Sigma"),
        ]

        self.paste_field_names = [f for _, f in self.infotext_fields]

        return [
            enabled,
            scale,
            mask_threshold,
            show_simmap,
            auto_th,
            blur_size,
            blur_sigma,
        ]

    def process(self, p: StableDiffusionProcessing, *args, **kwargs):
        (
            enabled,
            scale,
            mask_threshold,
            _,
            auto_th,
            blur_size,
            blur_sigma_,
        ) = args

        last_attn_masks.clear()
        last_sag_mask_thresholds.clear()

        global sag_enabled, sag_mask_threshold, sag_mask_threshold_auto
        global blur_kernel_size, blur_sigma
        if enabled:
            sag_enabled = True
            sag_mask_threshold = mask_threshold
            sag_mask_threshold_auto = auto_th
            blur_kernel_size = int(blur_size)
            blur_sigma = float(blur_sigma_)

            global current_sag_guidance_scale
            current_sag_guidance_scale = scale
            global saved_original_selfattn_forward
            # replace target self attention module in unet with ours

            org_attn_module = (
                shared.sd_model.model.diffusion_model.middle_block._modules["1"]
                .transformer_blocks._modules["0"]
                .attn1
            )
            saved_original_selfattn_forward = org_attn_module.forward
            org_attn_module.forward = xattn_forward_log.__get__(
                org_attn_module, org_attn_module.__class__
            )

            p.extra_generation_params["SAG Enabled"] = enabled
            p.extra_generation_params["SAG Guidance Scale"] = scale
            p.extra_generation_params["SAG Mask Threshold"] = mask_threshold
            p.extra_generation_params["SAG Auto Threshold"] = auto_th
            p.extra_generation_params["SAG Blur Kernel Size"] = blur_size
            p.extra_generation_params["SAG Blur Sigma"] = blur_sigma_

        else:
            sag_enabled = False

        if not hasattr(self, "callbacks_added"):
            on_cfg_denoiser(self.denoiser_callback)
            on_cfg_denoised(self.denoised_callback)
            on_cfg_after_cfg(self.cfg_after_cfg_callback)
            self.callbacks_added = True

        return

    def postprocess(self, p, processed, *args):
        enabled, scale, sag_mask_threshold, show_simmap, auto_th, *rest = args

        if enabled:
            # restore original self attention module forward function
            attn_module = (
                shared.sd_model.model.diffusion_model.middle_block._modules["1"]
                .transformer_blocks._modules["0"]
                .attn1
            )
            attn_module.forward = saved_original_selfattn_forward

            # add attention masks
            if show_simmap:
                for attn_mask in last_attn_masks:
                    B, C, H_lat, W_lat = attn_mask.shape
                    assert (
                        C == 4
                    )  # same as channels of U-Net output (each channel has same value)
                    attn_mask = attn_mask[:, 0, :, :]  # (B,H,W)
                    for b in range(B):
                        base_image = processed.images[
                            processed.index_of_first_image + b
                        ]
                        mask = attn_mask[b, :, :]
                        mask = torch.clamp((mask * 255), 0, 255).to(torch.uint8)
                        mask = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0),
                            size=(base_image.height, base_image.width),
                            mode="nearest",
                        )

                        mask_image = torchvision.transforms.ToPILImage("L")(
                            mask.squeeze()
                        ).convert("RGB")
                        image = PIL.Image.blend(base_image, mask_image, 0.5)

                        processed.images.append(image)

            if auto_th:
                print(
                    "SAG mask threshold:",
                    list(last_sag_mask_thresholds[1:]),
                    file=sys.stderr,
                )

        last_attn_masks.clear()
        last_sag_mask_thresholds.clear()


xyz_grid_support_sag.initialize(Script)
