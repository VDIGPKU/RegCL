from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file
from PIL import Image
import numpy as np
from skimage import io, transform
from icecream import ic
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
from augmodule_utils.utils import generate_heatmap, LayerNorm2d


class _LoRA_qkv(nn.Module):
    def __init__(self, r, attn: nn.Module):
        super().__init__()
        self.r = r
        self.attn = attn
        self.A = nn.Linear(768, r)
        self.B = nn.Linear(r, 768)
        self.cnt = self.A.weight.numel() + self.A.bias.numel() + self.B.weight.numel() + self.B.bias.numel()
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.A.bias)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def reset_zeros(self):
        nn.init.zeros_(self.A.weight)
        nn.init.zeros_(self.A.bias)

        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def add_parameters(self, model, ratio) -> None:
        new_weight = self.A.weight + model.A.weight * ratio
        self.A.weight = torch.nn.Parameter(new_weight)
        new_weight = self.A.bias + model.A.bias * ratio
        self.A.bias = torch.nn.Parameter(new_weight)

        new_weight = self.B.weight + model.B.weight * ratio
        self.B.weight = torch.nn.Parameter(new_weight)
        new_weight = self.B.bias + model.B.bias * ratio
        self.B.bias = torch.nn.Parameter(new_weight)

    def forward(self, x):
        attn = self.attn(x)
        adapter_ = self.A(x)
        adapter_ = self.B(adapter_)
        return attn + adapter_



class Adapter_Sam(nn.Module):
    def __init__(self, sam_model: Sam, r=10, lora_layer=None):
        super(Adapter_Sam, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(7, len(sam_model.image_encoder.blocks)))
        self.begin=self.lora_layer[0]
        self.r = r
        self.cnt = 0
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            blk.attn = _LoRA_qkv(r, blk.attn)
            self.cnt+= blk.attn.cnt

        self.sam = sam_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        lora_layer = {f"lora_layer": self.lora_layer}
        num_layer = len(self.lora_layer)
        A = {f"A_{i:03d}": self.sam.image_encoder.blocks[i+self.begin].attn.A for i in range(num_layer)}
        B= {f"B_{i:03d}": self.sam.image_encoder.blocks[i+self.begin].attn.B for i in range(num_layer)}

        merged_dict = {**lora_layer, **A, **B}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str, args) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        self.lora_layer = state_dict[f'lora_layer']
        i = 0
        for t_layer_i, blk in enumerate(self.sam.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            A = state_dict[f'A_{i:03d}']
            B = state_dict[f'B_{i:03d}']
            i+=1
            self.sam.image_encoder.blocks[t_layer_i].attn.A = A.cpu().cuda() #to(f'cuda:{args.rank}')
            self.sam.image_encoder.blocks[t_layer_i].attn.B = B.cpu().cuda() #to(f'cuda:{args.rank}')

    def reset_zeros(self):
        for t_layer_i, blk in enumerate(self.sam.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            self.sam.image_encoder.blocks[t_layer_i].attn.reset_zeros()

    def add_parameters(self, model, ratio) -> None:
        for t_layer_i, blk in enumerate(self.sam.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            self.sam.image_encoder.blocks[t_layer_i].attn.add_parameters(model.sam.image_encoder.blocks[t_layer_i].attn, ratio)

    def count_param(self):
        print("USE Param: ", self.cnt, " Memory: ", self.cnt*32/8/1024/1024, "MB")

    def forward(self, batched_input, multimask_output=False, image_size=1024, boxes=None, points=None, begin=-1, end=-1):
        x = batched_input
        if begin == -1:
            input_images = self.sam.preprocess(batched_input)
            input_images = self.sam.image_encoder.patch_embed(input_images)
            if self.sam.image_encoder.pos_embed is not None:
                input_images += self.sam.image_encoder.pos_embed
            x = input_images
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points, boxes=boxes, masks=None
        )
        for t_layer_i, blk in enumerate(self.sam.image_encoder.blocks):
            if t_layer_i < begin:
                continue
            x = blk(x)
            if t_layer_i == end:
                embed = x.detach()
                embed = embed.permute(0, 3, 1, 2)
                embed = embed.detach().mean(-1).mean(-1).tolist()
                return x, embed

        image_embeddings = self.sam.image_encoder.neck(x.permute(0, 3, 1, 2))
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        low_res_masks, iou_predictions, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.sam.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
        return outputs
