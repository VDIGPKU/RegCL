import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling import Sam
from augmodule_utils.utils import generate_heatmap


class _LoRA_qkv(nn.Module):
    def __init__(self, r, attn: nn.Module, A):
        super().__init__()
        self.r = r
        self.attn = attn
        self.A = A
        self.B = nn.Linear(r, 768)
        self.C = nn.Linear(r, r)
        self.norm1 = nn.LayerNorm(r)
        self.norm2 = nn.LayerNorm(r)
        self.cnt = self.B.weight.numel() + self.B.bias.numel() + self.C.weight.numel() + self.C.bias.numel()
        nn.init.kaiming_uniform_(self.C.weight, a=math.sqrt(5))
        nn.init.zeros_(self.C.bias)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, x, prompt_aug, interMed=False):
        attn = self.attn(x)
        adapter_ = self.A(x)
        if adapter_.size(0)!=prompt_aug.size(0):
            prompt_aug_ = F.interpolate(prompt_aug, size=(14, 14), mode='nearest')
            x_expanded = torch.cat([prompt_aug_[i].unsqueeze(0).repeat(int(adapter_.size(0)/prompt_aug.size(0)), 1, 1, 1) for i in range(prompt_aug.size(0))], dim=0).permute(0,3,2,1)
        else:
            x_expanded = prompt_aug.permute(0,3,2,1)
        adapter_ = self.C(self.norm1(adapter_) + self.norm2(x_expanded))
        adapter_ = self.B(adapter_)
        return attn + adapter_

class Extra(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.cnt = 0
        self.layer = nn.Linear(r, r)
        self.norm = nn.LayerNorm(r)
        self.upward = nn.Linear(r, 768)
        self.upward_ = nn.Linear(r, 256)
        self.cnt = self.layer.weight.numel() + self.layer.bias.numel() + self.upward.weight.numel() + self.upward.bias.numel() + self.upward_.weight.numel() + self.upward_.bias.numel()
        nn.init.zeros_(self.upward.weight)
        nn.init.zeros_(self.upward.bias)
        nn.init.zeros_(self.upward_.weight)
        nn.init.zeros_(self.upward_.bias)

class Adapter_Sam(nn.Module):
    def __init__(self, sam_model: Sam, r=10, lora_layer=None):
        super(Adapter_Sam, self).__init__()

        assert r > 0
        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(7, len(sam_model.image_encoder.blocks))) # loralayer num init og 7
        self.r = r
        self.cnt = 0
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False

        sam_model.image_encoder.train(mode=False)
        sam_model.mask_decoder.train(mode=False)
        sam_model.prompt_encoder.train(mode=False)

        self.extra = Extra(r)
        self.A = nn.Linear(768, 10)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.A.bias)
        self.cnt += self.extra.cnt + self.A.weight.numel() + self.A.bias.numel()

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            blk.attn = _LoRA_qkv(r, blk.attn, self.A)
            self.cnt+= blk.attn.cnt

        self.sam = sam_model


    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        extra = {f"extra": self.extra.cpu()}
        lora_layer = {f"lora_layer": self.lora_layer}
        start = self.lora_layer[0]
        num_layer = len(self.lora_layer)
        A = {f"A": self.A.cpu()}
        B= {f"B_{i:03d}": self.sam.image_encoder.blocks[i+start].attn.B.cpu() for i in range(num_layer)} # loralayer num init og 7
        C= {f"C_{i:03d}": self.sam.image_encoder.blocks[i+start].attn.C.cpu() for i in range(num_layer)} # loralayer num init og 7
        norm1 = {f"norm1_{i:03d}": self.sam.image_encoder.blocks[i+start].attn.norm1.cpu() for i in range(num_layer)} # loralayer num init og 7
        norm2 = {f"norm2_{i:03d}": self.sam.image_encoder.blocks[i+start].attn.norm2.cpu() for i in range(num_layer)} # loralayer num init og 7

        merged_dict = {**extra, **lora_layer, **A, **B, **C, **norm1, **norm2}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str, args) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        self.extra = state_dict[f'extra'].cpu().cuda()
        self.A = state_dict['A'].cpu().cuda()
        self.lora_layer = state_dict[f'lora_layer']
        i = 0
        for t_layer_i, blk in enumerate(self.sam.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            B = state_dict[f'B_{i:03d}']
            C = state_dict[f'C_{i:03d}']
            norm1 = state_dict[f'norm1_{i:03d}']
            norm2 = state_dict[f'norm2_{i:03d}']
            i+=1
            self.sam.image_encoder.blocks[t_layer_i].attn.A = self.A
            self.sam.image_encoder.blocks[t_layer_i].attn.B = B.cpu().cuda()#.to(f'cuda:{args.rank}')
            self.sam.image_encoder.blocks[t_layer_i].attn.C = C.cpu().cuda()#.to(f'cuda:{args.rank}')
            self.sam.image_encoder.blocks[t_layer_i].attn.norm1 = norm1.cpu().cuda()#.to(f'cuda:{args.rank}')
            self.sam.image_encoder.blocks[t_layer_i].attn.norm2 = norm2.cpu().cuda()#.to(f'cuda:{args.rank}')

    def count_param(self):
        print("USE Param: ", self.cnt, " Memory: ", self.cnt*32/8/1024/1024, "MB", "Extra:", self.extra.cnt*32/8/1024/1024, "MB")

    def forward(self, batched_input, multimask_output=False, image_size=1024, boxes=None, points=None, begin=-1, end=-1):
        # Generate the prompt heatmap
        if points is not None:
            # Use named constants instead of magic numbers
            HEATMAP_SIZE = (9, 9)
            HEATMAP_SCALE = (64, 64)

            prompt_heat = generate_heatmap(HEATMAP_SIZE, points, *HEATMAP_SCALE).cuda().float()
            prompt_heat = prompt_heat.repeat(1, self.r, 1, 1)
            prompt_heat = self.extra.layer(prompt_heat.permute(0,2,3,1))
            prompt_heat = self.extra.norm(prompt_heat)
            prompt_heat = F.relu(prompt_heat).permute(0,3,1,2)

        # Image preprocessing
        x = batched_input
        if begin == -1:
            # Run the standard preprocessing flow when no start layer is specified
            input_images = self.sam.preprocess(batched_input)
            input_images = self.sam.image_encoder.patch_embed(input_images)
            if self.sam.image_encoder.pos_embed is not None:
                input_images += self.sam.image_encoder.pos_embed
            x = input_images

        # Keep prompt encoding unchanged
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points, boxes=boxes, masks=None
        )

        # Iterate through encoder layers
        for t_layer_i, blk in enumerate(self.sam.image_encoder.blocks):
            if t_layer_i < begin:
                continue

            # Use a unified feature processing interface
            # x = blk(x, prompt_heat) if t_layer_i in self.lora_layer else blk(x)
            if t_layer_i in self.lora_layer:
                x = blk(x, prompt_heat)
            else:
                x = blk(x)

            if t_layer_i == end:
                embed = x.detach()
                embed = embed.permute(0, 3, 1, 2)
                embed = embed.detach().mean(-1).mean(-1).tolist()
                return x, embed

        prompt_heat_ = self.extra.upward(prompt_heat.permute(0,2,3,1)).permute(0,3,1,2)
        prompt_heat = self.extra.upward_(prompt_heat.permute(0,2,3,1)).permute(0,3,1,2)
        image_embeddings = self.sam.image_encoder.neck(x.permute(0, 3, 1, 2) + prompt_heat_) + prompt_heat

        # Keep mask decoding and postprocessing unchanged
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

        return {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
