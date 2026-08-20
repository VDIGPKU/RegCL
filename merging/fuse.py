import os
import torch
import copy
import argparse
import logging
from importlib import import_module
from segment_anything import sam_model_registry
from merging.inner_product import *
from train_regcl import get_dataloader_SAM, setup_seed, setup_distribution, load_dataset


def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_multi_LoRA_weights(net_origin, model_names) -> dict:
    """Load weights from multiple models."""
    params = {}
    for key in model_names: # load_model_prams
        x = copy.deepcopy(net_origin).cuda()  # Move the model to the target GPU
        x.load_lora_parameters(f'{args.model_folder}/{key}.pth', args)
        # x.load_lora_parameters(f'{args.model_folder}/ft_{key}.pth', args)
        print(f'model {key} loaded on {next(x.parameters()).device}')
        params.update({key: x})
    return params

def load_multi_dataloader(test_data, train_data, model_names, img_emb_size):
    loaders = {}
    for key in model_names:
        test_location = test_data[key]
        train_location= train_data[key]
        print('Loading test and train datasets...')
        print(f'getting dataloader for {key}')
        trainloader, _ = get_dataloader_SAM(args, img_emb_size, train_location, test_location)
        loaders.update({key: trainloader})
    return loaders

def check(params):
    """Print the number of trainable parameters."""
    for layer_name, layer in params.items():
        if isinstance(layer, torch.nn.Module):  # Ensure the layer is a module.
            for name, param in layer.named_parameters():
                if param.requires_grad:  # Only print parameters that require gradients
                    print(f'layer: {layer_name} - para_name: {name}')
                    print(f'shape: {param.shape}')
        else:
            print(f'el layer: {layer_name} - type: {type(layer)} - data: {layer}')

def fuse_weights(args, net, params_name, _weights=None):
    """
    Fuse model weights.
    :param weights_list: List of weights from multiple models.
    :param fusion_method: Fusion method, one of 'mean', 'weighted', or 'RegMean'.
    :param fusion_weights: Required weight list when fusion_method is 'weighted'.
    :return: Fused weight dictionary.
    """
    if not params_name:
        raise ValueError("params_list is empty!")

    param_dict = load_multi_LoRA_weights(net, params_name)
    param_lst = [param_dict[key] for key in params_name]

    lora_layer = param_lst[0].lora_layer
    print(f'LoRA layer: {lora_layer}')

    filterd_layer_inputs = [f'sam.image_encoder.blocks.{i}.attn.B' for i in lora_layer] + \
                           [f'sam.image_encoder.blocks.{i}.attn.C' for i in lora_layer] + \
                           [f'A']

    filterd_layer_params = [
                            '.*extra.*',
                            '.*A\.',
                            '.*\.attn\.norm1.*',
                            '.*\.attn\.norm2.*',
                            '.*\.attn\.C.*',
                            '.*\.attn\.B.*'
                            ]

    if args.method == "mean":
        print(f'Method: mean')

        fused_weights = copy.deepcopy(param_lst[0].state_dict())

        for key in fused_weights.keys():
            if any(re.match(pattern, key) for pattern in filterd_layer_params):
                print(f"Avg: {key} / {len(param_lst)}")
                for model in param_lst[1:]:
                    fused_weights[key] += model.state_dict()[key]
                fused_weights[key] /= len(param_lst)

    elif args.method == "RegMean":
        print(f'Method: RegMean')
        train_data, test_data = load_dataset(args)
        dataloader_dict = load_multi_dataloader(test_data, train_data, params_name, args.size)

        grams_dict = {}
        for key in params_name:
            with torch.no_grad():
                print(f'{key} gram computing...')
                grams = compute_gram(param_dict[key], dataloader_dict[key], filterd_layer_inputs)
            if grams is None:
                print(f'compute {key} gram Failed')
            else:
                print(f'compute {key} gram Done')
            grams_dict.update({key: grams})
        grams_lst = [grams_dict[key] for key in params_name]

        fused_weights = avg_merge(param_lst, grams_lst, filterd_layer_params) # actual computation

    else:
        raise ValueError(f"Unsupported fusion method: {args.method}")

    return fused_weights

def get_available_filename(base_path, method):
    """Get an available filename by incrementing the suffix automatically."""
    index = 0
    while True:
        filename = f"{base_path}/{method}_params_{index}.pth" if index > 0 else f"{base_path}/{method}_params.pth"
        if not os.path.exists(filename):
            return filename
        index += 1

def main(args):
    models_order_lst = args.order.split("_")
    output_path = os.path.join(args.model_folder, f"fuse_{'_'.join(models_order_lst)}")

    # Ensure the output path exists
    os.makedirs(output_path, exist_ok=True)

    setup_seed(args.seed)
    setup_distribution(args)

    # Load the model
    sam, args.size = sam_model_registry['vit_b'](checkpoint=args.ckpt)
    sam = sam.cuda()  # Move the model to the target GPU
    print(f'sam loaded on {next(sam.parameters()).device}')
    for _, param in sam.image_encoder.named_parameters():
        param.requires_grad = False
    sam.image_encoder.train(mode=False)

    start = int(args.layers.split('-')[0])
    end = int(args.layers.split('-')[1])
    lora_layer = list(range(start, end + 1))
    pkg = import_module(f'module.{args.module}')
    net_origin = pkg.Adapter_Sam(copy.deepcopy(sam), lora_layer=lora_layer)

    try:
        fused_params = fuse_weights(args, net_origin, models_order_lst)
        merged_model = copy.deepcopy(net_origin).cuda()
        copy_params_to_model(fused_params, merged_model)

        # Get an available filename
        save_path = get_available_filename(output_path, args.method)
        merged_model.save_lora_parameters(save_path)
        print(f'merged_model.lora_layer: {merged_model.lora_layer}')
        print(f"Model parameters saved to: {save_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu (Default=2)')
    parser.add_argument('--module', type=str,
                        default='AugModule', choices=['AugModule', 'LoRA'], help='Module (Default=AugModule)')
    parser.add_argument('--model_folder', type=str,
                        default='log/AugModule_Kvasir_camo_ISTD_ISIC_cod__00', help='Path to the models\' folder')
    parser.add_argument('--ckpt', type=str,
                        default='checkpoint/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
    parser.add_argument('--method', type = str,
                        default='RegMean', help='Fusion method mean/weighted/RegMean (Defalut=RegMean)')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed (Default=1024)')
    parser.add_argument('--order', type=str,
                        default="Kvasir_camo_ISTD_ISIC_cod", help="Training order (Default=Kvasir_camo_ISTD_ISIC_cod)")
    parser.add_argument('--cuda', type=int,
                        default=-1, help='ID of GPU when using single GPU (cuda=-1 means using distributed GPU)')
    parser.add_argument('--layers', type=str,
                        default='7-11', help='LoRA layers (Default=7-11)')
    args = parser.parse_args()

    main(args)
