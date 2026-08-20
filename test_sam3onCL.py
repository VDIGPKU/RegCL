import os
import json
import copy
import random
import logging
import argparse
import numpy as np
from importlib import import_module

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from segment_anything import sam_model_registry
from datasets.dataset import SAM_dataset, RandomGenerator
from augmodule_utils.utils import AverageMeter, mae, select

def setup_logging(args) -> logging.Logger:
    """Set up the log file and format."""
    # Generate the log filename
    if os.path.isfile(args.lora_path):
        log_name = os.path.basename(args.lora_path).replace('.pth', '_test.txt')
        log_file = os.path.join(os.path.dirname(args.lora_path), log_name)
    elif os.path.isdir(args.lora_path):
        log_name = 'log_test.txt'
        log_file = os.path.join(args.lora_path, 'log_test.txt')
    print(log_file)
    # Use a flexible logger configuration
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def setup_seed(seed: int = 1234):
    """Initialize random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def get_dataloader_SAM(args, img_embedding_size, test_data_location):
    """Create the DataLoader for the test dataset."""
    low_res = img_embedding_size * 4
    db_test = SAM_dataset(
        test_data_location,
        transform=transforms.Compose([
            RandomGenerator(output_size=[1024, 1024],
                            low_res=[low_res, low_res],
                            bbox_shift=20,
                            get_point=3)
        ]),
        inp_size=1024,
        type='test'
    )
    print(f"The length of test set is: {len(db_test)}")
    return DataLoader(db_test, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)

def evaluate_model(args, logging, testloader, sam, net, start):
    """Evaluate model performance."""
    ious, f1_scores, mae_scores = AverageMeter(), AverageMeter(), AverageMeter()
    net.eval()
    for iter, data in enumerate(testloader):
        images = data["image"].cuda(non_blocking=True)
        gt_masks = data["label"].cuda(non_blocking=True)
        points = [p.cuda(non_blocking=True) for p in data['point']]

        with torch.no_grad():
            if start > 0:
                end_before_lora = start - 1
                input_images = sam.preprocess(images)
                mid_embed, embed = sam.image_encoder(input_images, False, begin=-1, end=end_before_lora)
                outputs = net(mid_embed, points=points, begin=start, end=-1)
            else:
                outputs = net(images, points=points)

        for image_, pred_mask, gt_mask in zip(images, outputs["masks"], gt_masks):
            if len(gt_mask.size()) < 3:
                gt_mask = gt_mask.unsqueeze(0)
            batch_stats = smp.metrics.get_stats(
                torch.sigmoid(pred_mask),
                gt_mask.int(),
                mode='binary',
                threshold=0.5,
            )
            ious.update(smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise"), 1)
            f1_scores.update(smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise"), 1)
            mae_scores.update(mae(pred_mask, gt_mask), 1)

        if logging and iter % 50 == 0:
            logging.info(
                f'Val: [{iter}/{len(testloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- MAE: [{mae_scores.avg:.4f}]'
            )

    if logging:
        logging.info(
            f'Final Results: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- MAE: [{mae_scores.avg:.4f}]'
        )
    return ious.avg, f1_scores.avg, mae_scores.avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default='AugModule', choices=['AugModule', 'LoRA'], help='Module (Default=AugModule)')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select one vit model (Default=vit_b)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
    parser.add_argument('--img_size', type=int, default=1024, help='input patch size of network input (Default=1024)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (Default=1024)')
    parser.add_argument('--order', type=str, default="Kvasir_camo_ISTD_ISIC_cod", help="Training order (Default=Kvasir_camo_ISTD_ISIC_cod)")
    parser.add_argument('--cuda', type=int, default=-1, help='ID of GPU when using single GPU')
    parser.add_argument('--lora_path', type=str, default=None, help='LoRA model path')
    parser.add_argument('--layers', type=str, default='7-11', help='LoRA layers (Default=7-11)')
    args = parser.parse_args()

    setup_seed(args.seed)

    torch.cuda.set_device(args.cuda)
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"CUDA Index : {current_device}")
    print(f"Device Name: {device_name}")

    args.rank = 0
    print('Not using distributed mode')

    with open('datasets/datasets_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    keys = args.order.split("_")
    missing_keys = [key for key in keys if key not in test_data]
    if missing_keys:
        raise KeyError(f"Missing dataset keys in test config: {missing_keys}")
    test_data = {key: test_data[key] for key in keys}

    print(f'Order is {test_data.keys()}')

    logging = setup_logging(args)
    logging.info(str(args))

    sam, img_embedding_size = sam_model_registry[args.vit_name](checkpoint=args.ckpt)
    sam = sam.cuda()
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    sam.image_encoder.train(mode=False)

    start = int(args.layers.split('-')[0])
    end = int(args.layers.split('-')[1])
    lora_layer = list(range(start, end + 1))
    logging.info(f'Loading {args.module} module ...')
    pkg = import_module(f'module.{args.module}')
    net_origin = pkg.Adapter_Sam(copy.deepcopy(sam), lora_layer=lora_layer)

    for key, data_location in test_data.items():
        logging.info(f"------Dataset {key} is begin-------")
        testloader = get_dataloader_SAM(args, img_embedding_size, data_location)
        if os.path.isfile(args.lora_path):
            net = select(args, net_origin, args.lora_path)
        elif os.path.isdir(args.lora_path):
            pth = os.path.join(args.lora_path, f'{key}.pth')
            if not os.path.exists(pth):
                logging.info(f"------Model {key} does not exist-------")
                continue
            net = select(args, net_origin, pth)
            logging.info(f"------Model {key} is loaded-------")
        evaluate_model(args, logging, testloader, sam, net, start)

if __name__ == '__main__':
    main()
